import os

import streamlit as st
from zotero.zotero_search import load_zotero_library, build_zotero_index
from agents.agent import run_agent
from agents.tools import tool_executor
from memory.save_conversation import save_conversation, persist_image, image_caption

# 全局变量
# if "current_title" not in st.session_state:
#     st.session_state.current_title = None
# if "current_chunks" not in st.session_state:
#     st.session_state.current_chunks = None
# if "current_index" not in st.session_state:
#     st.session_state.current_index = None
# 聊天对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 多论文集合:
if "loaded_papers" not in st.session_state:
    # 结构: { "论文标题": {"chunks": [...], "index": faiss_index} }
    st.session_state.loaded_papers = {}

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

# 工具名称 → 中文状态提示的映射。
# agent.py 在真正执行工具前会先 yield 一个 __TOOL_STATUS__ 标记，
# 这里把它翻译成用户能看懂的进度文字；不在表里的工具会退化成"正在执行 xxx..."。
# ML 工具有两项明显比别的慢（要起子进程、导入 xgboost / 算 SHAP），
# 所以提示语里写清楚"需启动推理进程"，避免用户以为卡死。
TOOL_STATUS_MAP = {
    "open_zotero_paper":        "📖 正在 Zotero 中检索并加载论文...",
    "list_zotero_collections":  "🗂 正在读取 Zotero 分类结构...",
    "open_zotero_collection":   "📚 正在批量加载分类下的论文（首次较慢，请稍候）...",
    "retrieve_current_paper":   "🔍 正在检索论文相关内容...",
    "search_online_literature": "🌐 正在网络搜索相关文献...",
    "list_ml_models":           "🧠 正在读取可用模型...",
    "get_model_input_schema":   "📋 正在读取模型输入规范...",
    "predict_with_model":       "📈 正在调用模型预测...",
    "explain_prediction":       "🔬 正在计算 SHAP 解释（需启动推理进程）...",
}

# 显示历史消息
# 图片必须和历史文本一起重绘：chat_history 里 assistant 条目除 content 外还带 images。
# 只靠 markdown 渲染是不够的 —— st.markdown 不会把本地绝对路径的 ![](...) 显示成图片，
# 必须用 st.image 逐张画。这就是"下一轮看不到上一轮 SHAP 图"的根因。
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown( message["content"])
        for image in message.get("images") or []:
            path = image.get("path")
            if path and os.path.isfile(path):
                st.image(path, caption=image.get("caption", "SHAP 解释（瀑布图）"))
            else:
                st.caption("（此前生成的图片已丢失：{}）".format(path))

# 提取用户问题
user_input = st.chat_input("请输入要阅读的zotero文献\网络论文检索\调用机器学习模型...")

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

        # 每轮重置：本轮预测产生的 SHAP 图。
        # tools.py 的 explain_prediction 只把图片路径塞进这个列表（工具无法返回图片），
        # 真正的渲染放在流式输出结束之后，见下方 st.image 循环。
        st.session_state.shap_plots = []

        for token in run_agent(
            user_input, 
            my_tool_executor,
            chat_history=st.session_state.chat_history,
            loaded_titles=list(st.session_state.loaded_papers.keys())
        ):
            if token.startswith("__TOOL_STATUS__:"):
                # 工具执行状态标记 —— 显示 spinner 提示，不写入答案
                payload = token.split(":", 1)[1]
                tool_name, _, detail = payload.partition("|")
                status_text = TOOL_STATUS_MAP.get(tool_name, f"正在执行 {tool_name}...")
                if detail:
                    status_text += f"「{detail}」"
                status_placeholder.info(f"{status_text}")
            else:
                # 真正的回答 token —— 清除状态提示，流式写入
                status_placeholder.empty()
                answer_tokens.append(token)
                stream_placeholder.markdown("".join(answer_tokens))

        answer = "".join(answer_tokens)

        # 展示本轮 SHAP 解释图，并把它归档到 memory/images/ 后记进对话历史。
        # 放在回答之后而不是工具执行时：st.image 必须挂在当前的 chat_message 容器里，
        # 而工具执行发生在 run_agent 内部，拿不到这个容器。
        #
        # 三个动作缺一不可：
        #   1) persist_image 把系统临时目录里的中转图复制到 memory/images/<日期>/，
        #      并删掉中转文件——临时目录会被系统清理，不复制历史里的图就成死链；
        #   2) st.image 立即画出来（本轮可见）；
        #   3) 把图片连同图注写进 chat_history 的 assistant 条目，
        #      这样下次 rerun 重绘历史时还能看到（跨轮可见）。
        turn_images = []
        for plot in st.session_state.shap_plots:
            # 兼容两种写法：新的 dict 记录，以及早期直接存的路径字符串
            src = plot.get("path") if isinstance(plot, dict) else plot
            caption = image_caption(plot)
            if not src or not os.path.isfile(src):
                st.caption(f"SHAP 图生成失败：{src}")
                continue

            saved_path, md_rel = persist_image(src)
            st.image(saved_path or src, caption=caption)

            # 归档成功后删掉中转文件：长期副本已经在 memory/images/ 里了。
            # 必须先 st.image 再删（此处顺序即如此），避免读到已删除的路径。
            # 只有目标路径确实不同于源文件时才删 —— persist_image 失败会返回
            # (None, None)，那时 src 是唯一一份图，删掉就彻底没了。
            if saved_path and os.path.abspath(saved_path) != os.path.abspath(src):
                try:
                    os.remove(src)
                except OSError:
                    pass

            turn_images.append({
                # 存归档后的副本路径：session 里也要用稳定的那份
                "path": saved_path or src,
                "md_rel": md_rel,
                "caption": caption,
                "model_id": plot.get("model_id") if isinstance(plot, dict) else None,
                "target": plot.get("target") if isinstance(plot, dict) else None,
                "target_label": plot.get("target_label") if isinstance(plot, dict) else None,
                "unit": plot.get("unit") if isinstance(plot, dict) else None,
            })

    # images 一并写入 chat_history：重绘历史时据此恢复图片
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "images": turn_images,
    })
    # 保存历史对话到本地 Markdown 文件（图片同时归档到 memory/images/）
    save_conversation(user_input, answer, images=turn_images)
