from zotero.zotero_search import (
    open_zotero_paper,
    list_collections,
    get_collection_papers,
)
from zotero.zotero_pdf import get_pdf_path
from rag.literature_read import build_pipeline, retrieve_current_paper
from review.literature_review import search_online_literature
from rag.retriever import retrieve
from rag.embedding import model
from config import ZOTERO_MAX_COLLECTION_PAPERS, RETRIEVE_PER_PAPER_K

# ① 工具的 JSON Schema —— 告诉 LLM 每个工具叫什么、干什么、参数是什么
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_zotero_paper",
            "description": "打开用户自己 Zotero 文献库中【已收藏】的一篇论文，用于逐篇阅读和提问。仅当用户想阅读/打开其库中已有的某篇【单篇】论文时使用。若用户按 Zotero 分类（文件夹/子类/文库）来指代论文，请改用 open_zotero_collection。若用户想『查找/发现某个主题的相关论文』，请改用 search_online_literature。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "用于搜索论文的英文关键词或论文标题"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_zotero_collections",
            "description": "列出用户 Zotero 中【有论文的】全部分类（文件夹），返回每个分类的完整路径、论文总数与子分类数量。当用户按分类指代论文但名称不确定、或有多个同名分类时，先调用它来确认准确的分类路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "可选。只列出路径中包含该关键词的分类；不传则列出全部。"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_zotero_collection",
            "description": "按 Zotero 分类（文件夹）批量加载论文。传入分类名或路径（如『缺失值』或『机器学习/缺失值』），会连同其子分类一起取出该分类下的全部论文并一次性加载，用于多篇对比、文献综述。用户说『读取某分类/某个文件夹/我的文库下某个子类里的论文』时用这个工具，不要逐篇调用 open_zotero_paper。",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Zotero 分类名或路径，如『缺失值』、『机器学习/缺失值』、『我的文库/机器学习/缺失值』"
                    },
                    "include_subcollections": {
                        "type": "boolean",
                        "description": "是否连同子分类里的论文一起加载，默认 true。用户明确只要该分类本身时才传 false。"
                    }
                },
                "required": ["collection_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_current_paper",
            "description": "对已加载的论文进行语义检索，返回相关文本片段。可以指定检索某几篇论文，也可以检索所有已加载的论文（做多篇综述时不要传 paper_titles，以跨全部论文检索）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题"
                    },
                    "paper_titles": {
                        "type": "string",
                        "description": "要检索的论文标题，用英文逗号分隔。如果不传或传空，则检索所有已加载的论文。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_online_literature",
            "description": "在外部学术数据库（网上）按主题搜索并发现相关论文，返回标题、DOI 等信息。当用户想『查找/发现某领域或主题的相关论文』时使用。此工具不检索用户的 Zotero 文献库。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "用于在线搜索的英文学术关键词"
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]


def _format_collection_tree(collections):
    """把分类列表格式化成缩进的树状文本，供 LLM 与用户阅读。"""
    lines = []
    for col in sorted(collections, key=lambda c: c["path"]):
        depth = col["path"].count("/")
        lines.append(
            "{}{}（共 {} 篇，含 {} 个子分类）".format(
                "  " * depth, col["path"], col["total_papers"], col["subcollections"]
            )
        )
    return "\n".join(lines)


def _load_one_paper(session_state, paper):
    """加载单篇论文，返回 (title, status, detail)。

    status: "loaded" | "already" | "skipped"
    """
    title = paper["title"]
    if title in session_state.loaded_papers:
        return title, "already", ""

    pdf_path = get_pdf_path(paper["itemID"])
    if not pdf_path:
        return title, "skipped", "未找到可用的 PDF 附件"

    try:
        chunks, index = build_pipeline(pdf_path)
    except Exception as exc:  # 单篇失败不应拖垮整批
        return title, "skipped", "解析失败：{}".format(exc)

    session_state.loaded_papers[title] = {"chunks": chunks, "index": index}
    return title, "loaded", ""


def tool_executor(
    tool_name,
    args,
    session_state,
    papers,
    zotero_index
):

    # ① 打开Zotero论文（单篇）
    if tool_name == "open_zotero_paper":
        keyword = args["keyword"]
        results = open_zotero_paper(keyword, papers, zotero_index, top_k=1)

        if not results:
            return "未找到相关论文"

        paper = results[0]
        title = paper["title"]

        # 如果这篇论文已经加载过，直接跳过，不用重复处理 PDF
        if title in session_state.loaded_papers:
            loaded_titles = list(session_state.loaded_papers.keys())
            return (
                f"论文《{title}》已经加载过了，无需重复加载。\n"
                f"当前已加载的论文：{', '.join(loaded_titles)}\n"
                f"请调用 retrieve_current_paper 检索论文内容。"
            )

        pdf_path = get_pdf_path(paper["itemID"])
        if not pdf_path:
            return (
                f"论文《{title}》在 Zotero 中找不到可用的 PDF 附件，无法加载。\n"
                f"请告知用户该篇缺少 PDF，并询问是否改用其他论文。"
            )
        chunks, index = build_pipeline(pdf_path)
        # 累加写入，不覆盖其他论文
        session_state.loaded_papers[title] = {"chunks": chunks, "index": index}
        loaded_titles = list(session_state.loaded_papers.keys())

        return (
        f"论文已成功加载完毕。\n"
        f"标题：{title}\n"
        f"当前已加载的论文：{', '.join(loaded_titles)}\n"
        f"现在请调用 retrieve_current_paper 工具，传入用户的问题，来检索论文内容并作答。"
    )

    # ② 列出 Zotero 分类
    elif tool_name == "list_zotero_collections":
        keyword = (args.get("keyword") or "").strip() or None
        collections = list_collections(keyword=keyword)
        if not collections:
            return (
                "没有找到包含论文的分类"
                + (f"（筛选关键词：{keyword}）" if keyword else "")
                + "。请让用户确认分类名称，或改用 search_online_literature。"
            )
        return (
            f"共 {len(collections)} 个包含论文的分类（路径 / 论文总数）：\n"
            f"{_format_collection_tree(collections)}\n\n"
            "如果用户的表述与此处不一致，请先向用户确认应采用哪一个分类路径，"
            "再调用 open_zotero_collection。"
        )

    # ③ 按分类批量加载论文
    elif tool_name == "open_zotero_collection":
        name = args.get("collection_name", "")
        include_sub = args.get("include_subcollections", True)
        if include_sub is None:
            include_sub = True

        result = get_collection_papers(name, include_subcollections=bool(include_sub))

        if result["status"] != "ok":
            if result["status"] == "ambiguous":
                options = "\n".join("  - " + p for p in result["candidates"][:20])
                return (
                    f"分类「{name}」无法唯一确定，存在多个候选：\n{options}\n\n"
                    "请不要猜测。把候选列表原样告诉用户，请其指定其中一个完整路径，"
                    "然后再次调用 open_zotero_collection。"
                )
            # not_found：附上有论文的分类清单，便于用户挑
            collections = list_collections()
            options = "\n".join("  - " + c["path"] for c in collections[:30]) or "  （无）"
            return (
                f"未找到分类「{name}」。当前库中包含论文的分类如下：\n{options}\n\n"
                "请把最接近的候选告诉用户并请其确认；若确实没有，"
                "可改用 open_zotero_paper（按标题）或 search_online_literature。"
            )

        col = result["collection"]
        papers_found = result["papers"]

        if not papers_found:
            return (
                f"分类「{col['path']}」"
                + ("（含子分类）" if include_sub else "")
                + "下没有找到任何带论文条目的记录。请告知用户该分类为空。"
            )

        # 超过上限 → 不加载，先返回清单让用户确认（避免长时间等待）
        if len(papers_found) > ZOTERO_MAX_COLLECTION_PAPERS:
            breakdown = list_collections()
            prefix = col["path"] + "/"
            rows = [
                c for c in breakdown
                if c["path"] == col["path"] or c["path"].startswith(prefix)
            ]
            detail = "\n".join(
                "  - {}（{} 篇）".format(c["path"], c["total_papers"])
                for c in rows
            ) or "  （无子分类）"
            return (
                f"分类「{col['path']}」"
                + ("（含子分类）" if include_sub else "")
                + f"共 {len(papers_found)} 篇，超过单次加载上限 {ZOTERO_MAX_COLLECTION_PAPERS} 篇，"
                "因此【尚未加载任何论文】。\n"
                f"该分类的结构如下：\n{detail}\n\n"
                "请向用户说明这一点，并询问其选择：\n"
                "  (a) 指定上面某个子分类（篇数较少）继续加载；或\n"
                "  (b) 确认要加载全部，然后再调用一次本工具并说明用户已确认全部加载。\n"
                "在用户明确选择之前，不要加载，也不要改用 open_zotero_paper 逐篇加载。"
            )

        # 逐篇加载（失败隔离：单篇异常只跳过，不中断整批）
        loaded, already, skipped = [], [], []
        for i, paper in enumerate(papers_found, 1):
            print(
                f"[collection] 正在加载 {i}/{len(papers_found)}：{paper['title']}",
                flush=True,
            )
            title, status, detail = _load_one_paper(session_state, paper)
            if status == "loaded":
                loaded.append(title)
            elif status == "already":
                already.append(title)
            else:
                skipped.append((title, detail))

        all_loaded = list(session_state.loaded_papers.keys())
        lines = [
            f"分类「{col['path']}」处理完成"
            + (f"（含 {col['subcollections']} 个子分类）" if col["subcollections"] else ""),
            f"本次新加载 {len(loaded)} 篇，已在库中跳过 {len(already)} 篇，失败/跳过 {len(skipped)} 篇。",
        ]
        if loaded:
            lines.append("新加载：\n" + "\n".join("  - " + t for t in loaded))
        if already:
            lines.append("此前已加载：\n" + "\n".join("  - " + t for t in already))
        if skipped:
            lines.append(
                "跳过：\n" + "\n".join(f"  - {t}（{d}）" for t, d in skipped)
            )
        lines.append(f"当前已加载的全部论文（共 {len(all_loaded)} 篇）：")
        lines.append("\n".join("  - " + t for t in all_loaded))
        lines.append(
            "现在请调用 retrieve_current_paper（【不要】传 paper_titles，以跨全部已加载论文检索），"
            "然后综合这些论文作答；若用户要的是综述，请按「逐篇要点 → 方法对比 → 结论分歧 → 引用来源」组织。"
        )
        return "\n".join(lines)

    # ④ 当前论文进行RAG检索
    elif tool_name == "retrieve_current_paper":
        if not session_state.loaded_papers:
            return "当前没有打开任何论文，请先调用 open_zotero_paper 或 open_zotero_collection。"

        query = args["query"]
         # ✅ 解析要检索哪些论文
        titles_arg = args.get("paper_titles", "")
        if titles_arg:
            # LLM 传了逗号分隔的标题列表
            if isinstance(titles_arg, list):
                target_titles = [str(t).strip() for t in titles_arg]
            else:
                target_titles = [t.strip() for t in str(titles_arg).split(",")]
            target_titles = [t for t in target_titles if t]
        else:
            # 没传 → 检索所有已加载论文
            target_titles = list(session_state.loaded_papers.keys())

         # ✅ 从每篇目标论文中检索 top_k 条结果（按篇配额，保证每篇都被覆盖）
        all_results = []
        coverage = []
        for title in target_titles:
            paper_data = session_state.loaded_papers.get(title)
            if not paper_data:
                all_results.append(f"[未找到论文：{title}]")
                continue

            results = retrieve(
                query,
                model,
                paper_data["index"],
                paper_data["chunks"],
                k=RETRIEVE_PER_PAPER_K
            )
            coverage.append(f"  - 《{title}》：命中 {len(results)} 条")
            # 给每条检索结果标注来源论文
            for r in results:
                all_results.append(f"【来源：{title}】\n{r}")

         # ✅ 附加对话历史
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in session_state.chat_history[-6:]]
        )
        context = "\n\n".join(all_results)

        return f"""
=== Conversation History ===
{history_text}

=== Retrieval Coverage ===
检索范围：{len(target_titles)} 篇论文，每篇最多取 {RETRIEVE_PER_PAPER_K} 条
{chr(10).join(coverage) if coverage else "  （无）"}
若某篇命中 0 条，说明该篇可能未涉及此问题，回答时不要为它编造内容。

=== Retrieved Paper Content ===
{context}
"""
    
    # ⑤ 网络文献综述
    elif tool_name == "search_online_literature":
        result = search_online_literature(args["keyword"])
        return result
    
    return "未知工具"
