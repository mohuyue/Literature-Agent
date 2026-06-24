from zotero.zotero_search import open_zotero_paper
from zotero.zotero_pdf import get_pdf_path
from rag.literature_read import build_pipeline, retrieve_current_paper
from review.literature_review import search_online_literature

# ① 工具的 JSON Schema —— 告诉 LLM 每个工具叫什么、干什么、参数是什么
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_zotero_paper",
            "description": "从用户的 Zotero 文献库中搜索并打开一篇论文，用于后续阅读和提问。",
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
            "description": "对当前已打开的论文进行语义检索，返回相关文本片段和历史对话上下文（不生成最终答案）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description":"用户问题"
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
            "description": "在网上搜索相关学术文献，返回搜索结果。",
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

        results = open_zotero_paper(
            keyword,
            papers,
            zotero_index,
            top_k=1
        )

        if not results:
            return "未找到相关论文"

        paper = results[0]

        pdf_path = get_pdf_path(
            paper["itemID"]
        )

        chunks, index = build_pipeline(pdf_path)

        session_state.current_chunks = chunks
        session_state.current_index = index
        session_state.current_title = paper["title"]
        session_state.paper_history = []

        return (
    f"论文已成功加载完毕。\n"
    f"标题：{paper['title']}\n"
    f"现在请调用 retrieve_current_paper 工具，传入用户的问题，来检索论文内容并作答。"
)
    # ② 当前论文进行RAG检索
    elif tool_name == "retrieve_current_paper":

        if session_state.current_chunks is None:
            return "当前没有打开论文"

        answer = retrieve_current_paper(
            args["query"],
            session_state.current_chunks,
            session_state.current_index,
            session_state.paper_history
        )

        return answer
    # ③ 网络文献综述
    elif tool_name == "search_online_literature":

        result = search_online_literature(
            args["keyword"]
        )

        return result
    
    return "未知工具"