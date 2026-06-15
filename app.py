import streamlit as st
from zotero.zotero_search import load_zotero_library, build_zotero_index, search_zotero
from zotero.zotero_pdf import get_pdf_path
from rag.literature_read import build_pipeline, ask_pdf
from review.literature_review import literature_review
from agents.planner import plan

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

# 提取用户问题
user_input = st.chat_input("请输入要阅读的zotero文献\文献问题\想网上搜索的论文...")
# 显示历史消息
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown( message["content"])

# 收到输入后调用planner规划任务        
if user_input:
    # 记录历史对话
    st.session_state.chat_history.append({"role":"user", "content":user_input})

    with st.chat_message("user"):
        st.markdown(user_input)
    # 规划任务
    plan_result = plan(user_input)
    # 获取任务
    task = plan_result["task"]

    if task == "OPEN_ZOTERO":
        keyword = plan_result["keyword"]
        results = search_zotero(
            keyword,
            papers,
            zotero_index,
            top_k=1
        )

        paper = results[0]
        if not results:
            st.error("未找到相关文献，请将文献导入到zotero中")
        
        pdf_path = get_pdf_path(paper["itemID"])
        with st.spinner("正在加载论文..."):
            chunks, index = build_pipeline(pdf_path)

        st.session_state.current_chunks = chunks
        st.session_state.current_index = index
        st.session_state.current_title = paper["title"]
        st.session_state.paper_history = []
        answer = f"""
                📖 已打开论文：{paper["title"]}\n
                现在可以继续提问：
                - 总结这篇论文
                - 创新点是什么
                - 作者为什么采用该方法
                """
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
        
    elif task == "ASK": 
        if st.session_state.current_chunks is None:
            answer = "请先打开一篇论文"
        else:
            with st.spinner("思考中..."):
                with st.chat_message("assistant"):
                    answer =  st.write_stream(
                            ask_pdf(
                                plan_result["query"],
                                st.session_state.current_chunks,
                                st.session_state.current_index,
                                st.session_state.paper_history
                            )   )
            st.session_state.paper_history.append({"role":"user", "content":user_input})
            st.session_state.paper_history.append({"role":"assistant", "content":answer})
            st.session_state.chat_history.append({"role":"assistant", "content":answer})
            st.stop()

    elif task == "REVIEW":
        with st.spinner("搜索中..."):
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    literature_review(plan_result["keyword"]) )
        st.session_state.chat_history.append({"role":"assistant", "content":answer})
        st.stop()