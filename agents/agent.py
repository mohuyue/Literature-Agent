import json, re
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.tools import TOOL_SCHEMAS

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

SYSTEM_PROMPT = """
You are an academic literature agent.
You have access to the following tools:

1. open_zotero_paper
- Use ONLY when the user wants to READ one specific paper they already have in their own Zotero library.
- This loads exactly ONE paper. For several papers, prefer open_zotero_collection.

2. list_zotero_collections
- Lists the user's Zotero folders (collections) that contain papers, with full paths and counts.
- Use it when the user refers to a folder/collection but you are not sure of the exact name,
  or when open_zotero_collection reports an ambiguous / not-found collection.

3. open_zotero_collection
- Loads EVERY paper inside a Zotero folder in one call (recursively including subfolders by default).
- Use it whenever the user identifies papers by folder, e.g.
  "读取某分类下的论文", "我的文库里某个子类", "read all papers in the X collection".
- Do NOT load such papers one by one with open_zotero_paper.

4. retrieve_current_paper
- Use for ANY question about the content of paper(s) already loaded.
- Pass paper_titles (comma-separated) to target specific papers; omit it to search all loaded papers.
- For a multi-paper comparison or review, OMIT paper_titles so every loaded paper is searched.

5. search_online_literature
- Use when the user wants to FIND / DISCOVER papers on a topic, for example
  "查找…相关论文", "找关于 X 的文献", "search for papers about X".
- This queries external academic databases, NOT the user's Zotero library.
- 输出结果必须包含标题原文及doi

ROUTING — decide the tool by the user's intent:
- Discover / find papers on a topic                        -> search_online_literature
- Read one specific paper the user already has             -> open_zotero_paper
- Read papers identified by a Zotero FOLDER / collection   -> open_zotero_collection
  (if the folder name is uncertain or ambiguous            -> list_zotero_collections first)
- Answer a question about already-loaded papers            -> retrieve_current_paper
- When it is unclear whether the user means an online search or their own library, ASK the user first.

WORKFLOWS:
A) Reading a single Zotero paper:
  Step 1: open_zotero_paper
  Step 2: retrieve_current_paper with the user's question
  Step 3: synthesize a final answer in Chinese, citing the source paper

B) Reading a whole Zotero collection / doing a review (综述):
  Step 1: open_zotero_collection with the collection name or path
  Step 2: read its tool result carefully.
          - If it says the collection is over the size limit, it has NOT loaded anything:
            relay the sub-collection list to the user and ask whether to load everything
            or a specific sub-collection. Wait for the user's answer.
          - If it reports skipped papers, mention them to the user; do not pretend they were read.
  Step 3: retrieve_current_paper WITHOUT paper_titles, so all loaded papers are searched
  Step 4: answer using this structure when the user asked for a review:
          ① 逐篇要点 (one short paragraph per paper, with its exact title)
          ② 方法/结论对比 (compare across papers, point out agreements and disagreements)
          ③ 结论与空白 (what is settled, what remains open)
          ④ 来源 (list the exact paper titles used)
          Always attribute statements to the specific source paper title.

RULES:
- Invoke tools through the function-calling interface. Do NOT output JSON text like {"action": ...}.
- Never answer paper content questions without first calling retrieve_current_paper.
- Do NOT hallucinate paper content.
- If retrieval reports 0 hits for a paper, do not invent content for it.
- After receiving tool results, synthesize a final answer in Chinese.
"""

def run_agent(user_query: str, tool_executor, chat_history=None,
              loaded_titles=None, max_turns: int = 10):
    """
    完整的 Tool-Calling Agent 循环。
    参数:
        user_query:     用户当前问题
        tool_executor:  工具执行函数 (name, args) → result_str
        chat_history:   对话历史列表 [{"role": "user/assistant", "content": "..."}]
        loaded_titles:  已加载论文的标题列表（None 表示未加载任何论文）
        max_turns:      最大推理轮次
    使用流式，逐 token yield，实现打字机效果。
    """
    # ① 构建系统提示，注入当前论文状态
    if loaded_titles:
        paper_status = (
            f"\n\n【当前状态】已加载论文：{', '.join(loaded_titles)}\n"
            f"对于论文内容的问题，请直接调用 retrieve_current_paper，不要重复打开相同论文。\n"
            f"如需加载更多论文，请调用 open_zotero_paper（单篇）或 open_zotero_collection（整个分类）。\n"
        )
    else:
        paper_status = (
            "\n\n【当前状态】没有打开任何论文。"
            "如果用户按分类（文件夹/子类）指代论文，请调用 open_zotero_collection；"
            "如果用户指向某一篇具体论文，请调用 open_zotero_paper。"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT + paper_status}]

    # ② 注入对话历史（最近 3 轮，6 条消息）
    if chat_history:
        messages.extend(chat_history[-6:])

    # ③ 当前用户问题
    messages.append({"role": "user", "content": user_query})

    # agent循环：调用 LLM，判断是否要调工具；如果不调工具，则生成最终回答
    for _ in range(max_turns):
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
            stream=True
        )

        # 按 index 分别累加每个工具调用，避免多个并行调用的参数被拼成一个字符串
        tool_calls_buf = {}  # index -> {"id": str, "name": str, "args": str}

        # 接收流
        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            # 普通文本
            if delta.content:
                yield delta.content
            # 工具调用（按 index 分别累加）
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0) or 0
                    buf = tool_calls_buf.setdefault(
                        idx, {"id": "", "name": "", "args": ""}
                    )
                    if tc.id:
                        buf["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            buf["name"] = tc.function.name
                        if tc.function.arguments:
                            buf["args"] += tc.function.arguments

        # 本轮没有工具调用 → LLM 已给出最终回答
        if not tool_calls_buf:
            return

        # 逐个执行本轮的所有工具调用
        assistant_tool_calls = []
        tool_messages = []
        for idx in sorted(tool_calls_buf.keys()):
            call = tool_calls_buf[idx]
            name = call["name"]
            raw_args = call["args"] or "{}"
            try:
                args = json.loads(raw_args)
            except Exception as e:
                yield f"\n\n工具参数解析失败：{e}"
                return
            # 通知UI（附带目标名，便于把"正在批量加载分类「缺失值」"这类信息展示出来）
            detail = ""
            if isinstance(args, dict):
                detail = args.get("collection_name") or args.get("keyword") or ""
            detail = str(detail).replace("|", "/").strip()
            yield f"__TOOL_STATUS__:{name}|{detail}"
            # 执行工具
            result = tool_executor(name, args)

            assistant_tool_calls.append({
                "id": call["id"],
                "type": "function",
                "function": {"name": name, "arguments": raw_args},
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result),
            })

        # 把本轮所有工具调用与结果加入上下文，继续下一轮推理
        messages.append({"role": "assistant", "tool_calls": assistant_tool_calls})
        messages.extend(tool_messages)

    yield "（已达到最大推理轮次）"