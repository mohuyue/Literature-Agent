import json, re
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.tools import TOOL_SCHEMAS

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

SYSTEM_PROMPT = """
You are an academic literature agent.
You have access to the following tools:

1. open_zotero_paper
- ONLY use this when NO paper is currently loaded, AND the user wants to read a specific paper.
- After this tool returns successfully, you MUST immediately call retrieve_current_paper to answer the user's question. Do NOT call open_zotero_paper again.

2. retrieve_current_paper
- Use this for ANY question about the content of the currently opened paper.
- If a paper is already loaded, always start here. Never call open_zotero_paper again.

3. search_online_literature
- Use ONLY when the user explicitly asks to search the internet or find related external papers.
- 输出结果必须包含标题原文及doi

WORKFLOW for reading a paper:
  Step 1: call open_zotero_paper (only if not already loaded)
  Step 2: call retrieve_current_paper with the user's question
  Step 3: synthesize the retrieved content into a final answer in Chinese

RULES:
- Never call open_zotero_paper more than once per conversation turn.
- Never answer paper content questions without first calling retrieve_current_paper.
- Do NOT hallucinate paper content.
- After receiving tool results, synthesize a final answer in Chinese.
"""

def _try_parse_json_tool_call(text):
    """
    降级方案：某些模型不使用 tool_calls 字段，而是输出 ReAct 风格的 JSON 文本。
    尝试从文本中提取 {"action": "...", "action_input": {...}} 格式的工具调用。
    """
    candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if "action" in data and "action_input" in data:
                return data["action"], data["action_input"]
        except (json.JSONDecodeError, TypeError):
            continue
    return None

def run_agent(user_query: str, tool_executor, chat_history=None,
              current_title=None, max_turns: int = 3):
    """
    完整的 Tool-Calling Agent 循环。
    参数:
        user_query:     用户当前问题
        tool_executor:  工具执行函数 (name, args) → result_str
        chat_history:   对话历史列表 [{"role": "user/assistant", "content": "..."}]
        current_title:  当前已打开论文的标题（None 表示未打开）
        max_turns:      最大推理轮次
    使用流式，逐 token yield，实现打字机效果。
    """
    # ① 构建系统提示，注入当前论文状态
    if current_title:
        paper_status = f"\n\n【当前状态】已打开论文：《{current_title}》\n对于论文内容的问题，请直接调用 retrieve_current_paper，不要重复打开论文。"
    else:
        paper_status = "\n\n【当前状态】没有打开任何论文。如果用户想阅读论文，请先调用 open_zotero_paper。"

    messages = [{"role": "system", "content": SYSTEM_PROMPT + paper_status}]

    # ② 注入对话历史（最近 3 轮，6 条消息）
    if chat_history:
        # for msg in chat_history[-6:]:
        #     messages.append({"role": msg["role"], "content": msg["content"]})
        messages.extend(chat_history[-6:])

    # ③ 当前用户问题
    messages.append({"role": "user", "content": user_query})

    # agent循环：调用 LLM，判断是否要调工具；如果不调工具，则生成最终回答
    for _ in range(max_turns):
        # 非流式：判断 LLM 是否要调工具
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
            stream=True
        )
        assistant_text = ""
        tool_call_id = None
        tool_name = None
        tool_args = ""
        # 接收流
        for chunk in stream:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta
            # 普通文本
            if delta.content:
                assistant_text += delta.content
                yield delta.content
            # 工具调用
            if delta.tool_calls:
                tc = delta.tool_calls[0]
                if tc.id:
                    tool_call_id = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_name = tc.function.name
                    if tc.function.arguments:
                        tool_args += tc.function.arguments
        # 本轮结束
        if tool_name:
            try:
                args = json.loads(tool_args)
            except Exception as e:
                yield f"\n\n工具参数解析失败：{e}"
                return
            # 通知UI
            yield f"__TOOL_STATUS__:{tool_name}"
            # 执行工具
            result = tool_executor(tool_name, args)
            # 把工具结果作为新消息加入上下文
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    }
                ]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(result)
            })
            # 下一轮继续推理
            continue
        # 没有工具调用
        return
    
    yield "（已达到最大推理轮次）"