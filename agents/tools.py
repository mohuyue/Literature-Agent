import json

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
from ml import client as ml_client
from ml.client import MLError

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
    },
    # ---- 机器学习模型（4 个工具）----------------------------------------
    # 这 4 个工具构成一条"状态传递链"：
    #   list_ml_models        选模型
    #   get_model_input_schema 拿到变量名/单位/范围（values 的键必须来自这里）
    #   predict_with_model    用完全相同的变量名做预测
    #   explain_prediction    用完全相同的 values 做 SHAP
    # 工具描述是 LLM 行为的直接约束，所以"必须先把变量清单给用户"这类要求
    # 写在 description 里，而不是只写进 system prompt —— 它离 tool_call 的生成更近。
    {
        "type": "function",
        "function": {
            "name": "list_ml_models",
            "description": "列出本地可用的已训练机器学习模型：每个模型能预测哪些性能指标、需要几个必填输入变量，以及它属于该模型族的第几代。注意：这些模型预测同样的性能指标，是同一技术路线按 1→2→3 代递进的关系，不是等价候选；用户提到『用模型预测/估算某性能』但未指明模型时，先调用它，再按用户手上的数据选代际。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_input_schema",
            "description": "获取某个模型的输入变量规范：必填变量（中文名、单位、取值范围）、可选变量、可预测目标与其精度，以及哪些分子量可以用特性粘度替代。调用后【必须】把必填变量清单完整呈现给用户并等待用户提供数值；若某些分子量项标注了『也可直接给 η_xxx 代替』，要把这个选项一并告诉用户（实验上特性粘度更好测）。严禁自行编造、猜测或使用默认值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "模型 ID，来自 list_ml_models 的返回值"
                    }
                },
                "required": ["model_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_with_model",
            "description": "用户提供输入变量取值后，调用机器学习模型预测性能指标。第 2、3 代模型的 values 里可以直接给分子量（Mv_*），也可以用特性粘度（eta_sol / eta_insol / eta_total）替代，系统会按修正 Mark-Houwink 方程自动换算；第 1 代模型（XGB_experimental_features）的 η_* 本身就是输入特征，不做换算、也不能用 Mv_* 替代。若返回『缺少必填变量』，请把这些变量告知用户并等待补充，不要自己填数，也不要自己换算粘度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "模型 ID"
                    },
                    "values": {
                        "type": "object",
                        "description": "变量名 → 数值 的映射，例如 {\"E_sol\": 24, \"E_insol\": 3.8}。键必须来自 get_model_input_schema 返回的变量名。分子量项 Mv_* 与特性粘度项 eta_* 二选一即可，无需同时提供。",
                        "additionalProperties": True
                    },
                    "targets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选。只想预测部分目标时传入目标名列表（如 [\"IS\"]）；不传则预测全部。"
                    }
                },
                "required": ["model_id", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_prediction",
            "description": "用 SHAP 解释某次预测：给出基准值、各变量对结果的贡献（正负号表示推高或拉低）。用户问『为什么是这个结果』『哪个因素影响最大』『怎么调整』时调用。必须使用与 predict_with_model 相同的 values 和相同的 model_id——三代模型的输入特征集合不同，混用会让解释失去意义。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "模型 ID"
                    },
                    "values": {
                        "type": "object",
                        "description": "与预测时相同的输入变量映射",
                        "additionalProperties": True
                    },
                    "target": {
                        "type": "string",
                        "description": "要解释的目标名，如 IS / FM / HZ / LT"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回贡献最大的前几个变量，默认 8"
                    }
                },
                "required": ["model_id", "values", "target"]
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


# ---------------------------------------------------------------------------
# 机器学习工具的结果格式化
# ---------------------------------------------------------------------------
def _fmt_num(value):
    """把数值格式化成人能读的形式（工具结果最终要给 LLM 和用户看）。

    分子量这类量级（19 万）用千分位；十亿以上（Mv_var）退回科学计数，
    否则一行数字会过长、反而难读。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    # 分子量量级用千分位更易读；十亿以上才退回科学计数，避免一行过长
    if v != 0 and abs(v) >= 1e9:
        return "{:.4g}".format(v)
    if abs(v) >= 1e4:
        return "{:,.0f}".format(v)
    return "{:g}".format(round(v, 4))


def _fmt_range(item):
    """格式化训练范围。上下界必须用同一种记法，否则会出现
    "397,958,332 ~ 5.139e+09" 这种一眼看不出大小的混搭。"""
    rng = item.get("range")
    if not rng or len(rng) != 2:
        return "未记录"
    low, high = float(rng[0]), float(rng[1])
    # 上下界用同一种记法，避免出现 "397,958,332 ~ 5.139e+09" 这种混搭
    if max(abs(low), abs(high)) >= 1e9:
        return "{:.4g} ~ {:.4g}".format(low, high)

    def bound(v):
        if abs(v) >= 1e4:
            return "{:,.0f}".format(v)
        return "{:g}".format(round(v, 4))

    return "{} ~ {}".format(bound(low), bound(high))


def _fmt_ml_models(models):
    """模型清单。刻意只列"能预测什么 / 要几个输入"——
    完整的变量清单留给 get_model_input_schema，避免第一步就把上下文塞满。

    但代际关系必须在这里讲清楚：三个模型预测同样的 4 个指标，只列"输入个数"
    会让 Agent 以为它们是等价候选、随便挑一个。所以按 1→2→3 排列并写明
    每一代"多了什么"，最后附选型规则。
    """
    if not models:
        return "本地没有可用的机器学习模型。"

    # 模型族说明与选型规则存在每份 metadata 里（内容相同），取第一份即可
    family_name = models[0].get("family_name") or ""
    try:
        meta0 = ml_client.get_metadata(models[0]["model_id"])
        family = meta0.get("family") or {}
        guide = meta0.get("selection_guide") or []
    except MLError:
        family, guide = {}, []

    lines = ["本地可用的机器学习模型（共 {} 个）".format(len(models))]
    if family_name:
        lines.append("它们属于同一模型族：{}，按代际递进，不是三个并列的候选。".format(family_name))
    if family.get("progression_note"):
        lines.append(family["progression_note"])

    for m in models:
        lines.append("")
        stage = m.get("stage")
        prefix = "第 {} 代｜".format(stage) if stage else ""
        lines.append("- {}{}｜{}".format(prefix, m["model_id"], m["display_name"]))
        if m.get("stage_name"):
            # stage_summary 是"这一代比上一代多了什么"的一行话，比 description 短
            lines.append("    定位：{}".format(m["stage_name"]))
        if m.get("stage_summary"):
            lines.append("    做法：{}".format(m["stage_summary"]))
        if m.get("description"):
            lines.append("    说明：{}".format(m["description"]))
        lines.append("    可预测：{}".format("；".join(m["targets"])))
        extra = ""
        if m.get("n_viscosity_alternatives"):
            # 提示 Agent：分子量可以不问，改问特性粘度（实验上更好测）
            extra = "（其中 {} 项分子量可用特性粘度替代）".format(
                m["n_viscosity_alternatives"])
        lines.append("    输入变量：必填 {} 个，可选 {} 个{}".format(
            m["n_required_inputs"], m["n_optional_inputs"], extra))
        if stage == 1:
            # 第 1 代的 η 是直接特征、不换算；不说清楚 Agent 会照搬第 2/3 代的说法
            lines.append("    注意：该模型的 η_* 是直接输入特征，不会换算成分子量；"
                         "若用户只有 Mv_* 而没有 η_*，本模型不可用。")

    if guide:
        lines += ["", "【如何选择】"]
        for g in guide:
            lines.append("  - " + g)

    return "\n".join(lines)


def _fmt_input_schema(meta):
    """把 metadata 渲染成给 LLM 看的变量清单（这是"Agent 会问用户什么"的直接来源）。

    三段结构：
      可预测目标   —— 让用户知道能拿到什么，附带 OOF 精度避免过度承诺
      必填变量     —— 必须完整呈现给用户，且标明单位与范围
      可选变量     —— BN 模型专属；说明"不提供会自动推断"，纯 XGB 模型这段为空
    末尾两行是行为约束，重复 system prompt 里的要求是有意为之：
    LLM 漏读一处很常见，两处都写能显著降低编造输入的概率。
    """
    lines = [
        "模型：{}（{}）".format(meta.get("display_name", ""), meta.get("model_id", "")),
        "",
    ]
    # 代际信息放在最前面：Agent 常需要回答"为什么用这个模型、和另外两个什么关系"
    family = meta.get("family") or {}
    if family.get("stage"):
        lines.append("代际：第 {} 代 —— {}（{}）".format(
            family["stage"], family.get("stage_name", ""), family.get("stage_summary", "")))
        lines.append("本代新增：{}".format(family.get("this_stage_adds", "-")))
        peers = [
            "第 {} 代 {}".format(x["stage"], x["model_id"])
            for x in family.get("lineage", []) if not x.get("is_self")
        ]
        if peers:
            lines.append("同族其它模型：{}（输入特征集合不同，不可与本模型混用）".format(
                "；".join(peers)))
        lines.append("")

    lines.append("【可预测目标】")
    for t in meta.get("targets", []):
        metrics = t.get("metrics") or {}
        extra = ""
        if metrics.get("r2_oof") is not None:
            extra = "，参考精度 R²={}、RMSE={}".format(
                metrics.get("r2_oof"), metrics.get("rmse_oof"))
        lines.append("  - {} {}（{}，训练范围 {}）{}".format(
            t.get("name"), t.get("label", ""), t.get("unit", "-"), _fmt_range(t), extra))

    required = meta.get("inputs", {}).get("required", [])
    optional = meta.get("inputs", {}).get("optional", [])

    lines += ["", "【必填变量】共 {} 个：".format(len(required))]
    for i, x in enumerate(required, 1):
        # alternatives 表示"这一项可以改填别的量"（目前只有分子量 → 特性粘度）
        alt = x.get("alternatives") or []
        alt_txt = "（也可直接给 {} 代替）".format("、".join(alt)) if alt else ""
        lines.append("  {}. {} —— {}，单位 {}，取值范围 {}{}".format(
            i, x["name"], x.get("label", ""), x.get("unit", "-"), _fmt_range(x), alt_txt))

    if optional:
        lines += ["", "【可选变量】共 {} 个（不提供则由模型自动推断；提供则覆盖推断值）：".format(
            len(optional))]
        for i, x in enumerate(optional, 1):
            alt = x.get("alternatives") or []
            alt_txt = "；也可直接给 {} 代替".format("、".join(alt)) if alt else ""
            lines.append("  {}. {} —— {}，单位 {}，取值范围 {}（由 {} 推断{}）".format(
                i, x["name"], x.get("label", ""), x.get("unit", "-"),
                _fmt_range(x), x.get("derived_by", "-"), alt_txt))

    # ---- 特性粘度替代路径 ----
    # 这段是"材料专业用户"最需要的信息：实验室往往先测得 η，分子量是换算出来的。
    # 明确写出方程与常数，Agent 才能在用户追问时解释换算依据，而不是含糊带过。
    visc = meta.get("viscosity_conversion") or {}
    rules = visc.get("rules", [])
    if visc.get("enabled") and rules:
        lines += [
            "",
            "【可替代输入：特性粘度】共 {} 项".format(len(rules)),
            "  用户若只有特性粘度（不是分子量），可以直接提供下列变量，系统会自动换算：",
        ]
        for r in rules:
            lines.append("  - {} —— {}（{}，取值范围 {}）→ 换算为 {}".format(
                r["eta"], r.get("eta_label", ""), r.get("eta_unit", "-"),
                _fmt_range({"range": r.get("eta_range")}), r["target"]))
        lines += [
            "  换算公式：{}".format(visc.get("equation", "modified Mark-Houwink")),
            "  其中 K_PE={}、alpha={}，W_E 为乙烯单元质量分数（即乙烯含量 wt% / 100）。".format(
                visc.get("K_PE"), visc.get("alpha")),
            "  各条规则的配套乙烯含量：{}。".format("，".join(
                "{} 用 {}".format(r["eta"], r["ethylene_input"]) for r in rules)),
            "  因此：问用户时把『分子量』和『对应的特性粘度』并列给出，任选其一即可；"
            "不要要求用户自己换算，也不要用你估算的分子量代替。",
        ]
    elif visc.get("note"):
        # 第 1 代模型：η 本身就是特征，不做换算。仍要告诉 Agent 这条边界，
        # 否则它会照搬"可以给 η 代替 Mv"的说法，而那在本模型里是错的。
        lines += ["", "【关于特性粘度】", "  " + visc["note"]]

    lines += [
        "",
        "请把上面的【必填变量】清单完整列给用户（包含单位与取值范围），然后等待用户提供取值。",
        "严禁自行编造、猜测或用默认值替代用户未提供的数值。",
    ]
    return "\n".join(lines)


def _fmt_prediction(data):
    """预测结果渲染。

    predictions 与 inferred 分开列：前者是答案，后者是"推断出的中间量"。
    中间量也返回给 LLM，是为了让它能回答"结晶度多少"这类追问，而不是自己编。
    warnings（外推告警）必须显式带上"结果不可信"，不能静默给个数字。
    """
    lines = ["预测完成（模型：{}）".format(data.get("model_name", "")), "", "【预测结果】"]
    for p in data.get("predictions", []):
        tag = "（用户给定值，非模型预测）" if p.get("provided") else ""
        lines.append("  {} {}：{} {}{}".format(
            p["name"], p.get("label", ""), _fmt_num(p["value"]), p.get("unit", ""), tag))

    # 换算过程必须回显：用户给的是 η，模型吃的是 Mv，
    # 不说明中间这一步，用户无法判断结果是否对得上自己的实验记录。
    conversions = data.get("conversions") or []
    if conversions:
        lines += ["", "【由特性粘度换算出的分子量】"]
        for c in conversions:
            source = "用户给定" if c.get("ethylene_source") == "user" else "由模型推断"
            lines.append("  {}={} {}（配套 {}={} wt%，{}）→ {}={} g/mol".format(
                c["eta"], _fmt_num(c["eta_value"]), "dL/g",
                c["ethylene_input"], _fmt_num(c["ethylene_value"]), source,
                c["target"], _fmt_num(c["mw"])))

    inferred = data.get("inferred") or []
    if inferred:
        lines += ["", "【由贝叶斯网络推断出的中间量】"]
        lines.append("  " + "、".join(
            "{}={}".format(x["name"], _fmt_num(x["value"])) for x in inferred))

    warnings = data.get("warnings") or []
    if warnings:
        lines += ["", "【外推告警】"]
        for w in warnings:
            lines.append("  ⚠ " + w)

    lines += ["", "如需解释成因（哪些变量推高/拉低了结果），请用相同的 values 调用 explain_prediction。"]
    return "\n".join(lines)


def _fmt_explanation(data):
    """SHAP 结果渲染。

    第一行刻意写成"预测值 = 基准值 + 贡献之和"，因为这是可检验的恒等式；
    用户能据此判断解释是否可信。inferred_parents 提醒"这些特征不是你填的"，
    否则容易被误解为归因到了用户输入。
    """
    unit = data.get("unit", "")
    pred = data.get("prediction", 0.0)
    base = data.get("base_value", 0.0)
    lines = [
        "SHAP 解释：{}（{}，{}）".format(
            data.get("target_label", ""), data.get("target", ""), unit),
        "预测值 {} = 基准值 {} + 各变量贡献之和 {}".format(
            _fmt_num(pred), _fmt_num(base), _fmt_num(pred - base)),
        "",
        "【各变量贡献】按影响绝对值排序，正号推高、负号拉低：",
    ]
    for c in data.get("contributions", []):
        lines.append("  {:<10} 取值 {:>12}  →  {:+8.3f} {}".format(
            c["name"], _fmt_num(c["value"]), c["shap"], unit))

    other = data.get("other_features_sum", 0.0)
    if abs(other) > 1e-9:
        lines.append("  其余变量合计：{:+.3f} {}".format(other, unit))

    inferred = data.get("inferred_parents") or []
    if inferred:
        lines += ["", "注意：{} 本身是由模型推断出来的中间量，不是用户直接输入的。".format(
            "、".join(inferred))]

    # 若某个父节点来自特性粘度换算，要说清它是"用你给的 η 算出来的"，
    # 否则用户会以为自己填过这个分子量。
    conversions = data.get("conversions") or []
    if conversions:
        lines += ["", "其中以下分子量由你提供的特性粘度换算得到（不是直接输入）："]
        for c in conversions:
            lines.append("  {} ← {}={} dL/g".format(
                c["target"], c["eta"], _fmt_num(c["eta_value"])))

    # 不把 plot_path 写进文本：那是系统临时目录里的中转文件，归档后立即删除，
    # 写出来只会让 Agent 对着一串临时路径念给用户听（图由 app.py 单独渲染）。
    # 条件判断不能省：画图失败时 plot_path 为 None，此时说"已生成"就是假报。
    if data.get("plot_path"):
        lines += ["", "（本轮已生成 SHAP 瀑布图，界面会直接显示，无需向用户描述图片路径。）"]
    return "\n".join(lines)


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

    # ⑥ 机器学习模型：列清单
    # 不需要起 worker 进程：纯读 metadata，因此这一步几乎没有延迟。
    elif tool_name == "list_ml_models":
        models = ml_client.list_models()
        if not models:
            return "本地没有可用的机器学习模型。请告知用户需要先运行 ML_model/export_models.py 导出模型。"
        return _fmt_ml_models(models)

    # ⑦ 机器学习模型：输入规范
    # 同样只看 metadata。返回的文本里包含变量名，是后续 values 字典键的唯一合法来源。
    elif tool_name == "get_model_input_schema":
        model_id = args.get("model_id") or ""
        try:
            meta = ml_client.get_metadata(model_id)
        except MLError as exc:
            # 转成文本返回而不是抛异常：让 LLM 看到"模型不存在，可用的是 X/Y"后自己纠正，
            # 比整轮对话崩掉更利于恢复。
            return str(exc)
        return _fmt_input_schema(meta)

    # ⑧ 机器学习模型：预测
    elif tool_name == "predict_with_model":
        model_id = args.get("model_id") or ""
        values = args.get("values") or {}
        # LLM 有时会把对象再序列化成字符串（"{\"E_sol\": 24}"），这里兜底解析一次。
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except Exception:
                return "values 必须是 JSON 对象（变量名→数值），收到的是字符串且无法解析。"
        if not isinstance(values, dict):
            return "values 必须是 JSON 对象（变量名→数值）。"
        try:
            data = ml_client.predict(model_id, values, args.get("targets"))
        except MLError as exc:
            # 错误文本会被原样交给 LLM。worker 在"缺少必填变量"时已经把
            # "请告知用户并等待补充，不要编造取值"写进了错误信息，
            # 所以这里不需要再拼一遍指令。
            return "模型预测失败：{}".format(exc)
        return _fmt_prediction(data)

    # ⑨ 机器学习模型：SHAP 解释
    elif tool_name == "explain_prediction":
        model_id = args.get("model_id") or ""
        values = args.get("values") or {}
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except Exception:
                return "values 必须是 JSON 对象（变量名→数值），收到的是字符串且无法解析。"
        if not isinstance(values, dict):
            return "values 必须是 JSON 对象（变量名→数值）。"
        target = args.get("target")
        if not target:
            return "缺少 target 参数，请指定要解释的目标（如 IS / FM / HZ / LT）。"
        try:
            data = ml_client.explain(
                model_id, values, target, int(args.get("top_k") or 8))
        except MLError as exc:
            return "模型解释失败：{}\n请把原因告知用户，不要编造解释。".format(exc)

        # 工具只能返回文本，图片没法直接塞给 LLM。
        # 所以这里把"图片路径 + 归属信息"存进 session_state，
        # 由 app.py 在本轮回答结束后统一 st.image() 并把图片并入对话历史。
        # 存 dict 而不是裸路径：历史面板要显示"这是哪个模型、哪个目标"的图注，
        # 只靠文件名反解既脆弱又难读。
        plot_path = data.get("plot_path")
        if plot_path:
            plots = getattr(session_state, "shap_plots", None)
            if plots is None:
                plots = []
                session_state.shap_plots = plots
            plots.append({
                "path": plot_path,
                # 用 worker 回传的规范模型名，而不是 LLM 传进来的原始参数
                "model_id": data.get("model_id") or model_id,
                "target": data.get("target"),
                "target_label": data.get("target_label"),
                "unit": data.get("unit"),
            })
        return _fmt_explanation(data)

    return "未知工具"
