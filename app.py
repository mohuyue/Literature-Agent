import streamlit as st
from zotero.zotero_search import load_zotero_library, build_zotero_index
from agents.agent import run_agent
from agents.tools import tool_executor
from memory.save_conversation import save_conversation

# 全局变量
if "current_title" not in st.session_state:
    st.session_state.current_title = None
if "current_chunks" not in st.session_state:
    st.session_state.current_chunks = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None
# 聊天对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# 论文阅读历史
if "paper_history" not in st.session_state:
    st.session_state.paper_history = []

# 页面标题
st.title("📚 Literature Reader Agent")

# 加载Zotero文献库数据
@st.cache_resource
def load_library():
    papers = load_zotero_library()
    index = build_zotero_index(papers)
    return papers, index

with st.spinner("请关闭你的zotero，正在连接..."):
    papers, zotero_index = load_library()

# 工具执行器
def my_tool_executor(name, args):
    return tool_executor(
        name,
        args,
        st.session_state,
        papers,
        zotero_index
    )

# 工具名称 → 中文状态提示的映射
TOOL_STATUS_MAP = {
    "open_zotero_paper":       "📖 正在 Zotero 中检索并加载论文...",
    "retrieve_current_paper":  "🔍 正在检索论文相关内容...",
    "search_online_literature": "🌐 正在网络搜索相关文献...",
}

# 显示历史消息
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown( message["content"])

# 提取用户问题
user_input = st.chat_input("请输入要阅读的zotero文献\文献问题\想网上搜索的论文...")

if user_input:
    # 立即显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 收到输入后调用 agent 规划任务
    with st.chat_message("assistant"):
        answer_tokens = []
        status_placeholder = st.empty()   # 用于显示/清除工具执行状态
        stream_placeholder = st.empty()   # 用于逐 token 更新答案

        for token in run_agent(
            user_input, 
            my_tool_executor,
            chat_history=st.session_state.chat_history,
            current_title=st.session_state.current_title
        ):
            if token.startswith("__TOOL_STATUS__:"):
                # 工具执行状态标记 —— 显示 spinner 提示，不写入答案
                tool_name = token.split(":", 1)[1]
                status_text = TOOL_STATUS_MAP.get(tool_name, f"正在执行 {tool_name}...")
                status_placeholder.info(f"⚙️ {status_text}")
            else:
                # 真正的回答 token —— 清除状态提示，流式写入
                status_placeholder.empty()
                answer_tokens.append(token)
                stream_placeholder.markdown("".join(answer_tokens))

        answer = "".join(answer_tokens)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    # 保存历史对话到本地 Markdown 文件
    save_conversation(user_input, answer)