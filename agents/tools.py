from zotero.zotero_search import open_zotero_paper
from zotero.zotero_pdf import get_pdf_path
from rag.literature_read import build_pipeline, retrieve_current_paper
from review.literature_review import search_online_literature
from rag.retriever import retrieve
from rag.embedding import model

# ① 工具的 JSON Schema —— 告诉 LLM 每个工具叫什么、干什么、参数是什么
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_zotero_paper",
            "description": "打开用户自己 Zotero 文献库中【已收藏】的一篇论文，用于逐篇阅读和提问。仅当用户想阅读/打开其库中已有的某篇论文时使用。若用户想『查找/发现某个主题的相关论文』，请改用 search_online_literature。",
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
            "name": "retrieve_current_paper",
            "description": "对已加载的论文进行语义检索，返回相关文本片段。可以指定检索某几篇论文，也可以检索所有已加载的论文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description":"用户问题"
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

def tool_executor(
    tool_name,
    args,
    session_state,
    papers,
    zotero_index
):

    # ① 打开Zotero论文
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

    # ② 当前论文进行RAG检索
    elif tool_name == "retrieve_current_paper":
        if not session_state.loaded_papers:
            return "当前没有打开任何论文，请先调用 open_zotero_paper。"
        
        query = args["query"]
         # ✅ 解析要检索哪些论文
        titles_arg = args.get("paper_titles", "")
        if titles_arg:
            # LLM 传了逗号分隔的标题列表
            target_titles = [t.strip() for t in titles_arg.split(",")]
        else:
            # 没传 → 检索所有已加载论文
            target_titles = list(session_state.loaded_papers.keys())
        
          # ✅ 从每篇目标论文中检索 top_k 条结果
        all_results = []
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
                k=3  # 每篇取 3 条，避免总量过大
            )
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

=== Retrieved Paper Content ===
{context}
"""
    
    # ③ 网络文献综述
    elif tool_name == "search_online_literature":
        result = search_online_literature(args["keyword"])
        return result
    
    return "未知工具"