from rag.pdf_loader import load_pdf
from rag.cleaner import clean_text
from rag.text_splitter import split_text
from rag.embedding import get_embeddings, model
from rag.vector_store import build_faiss_index
from rag.retriever import retrieve
from agents.llm import llm_stream

def build_pipeline(pdf_path):

    # 1. 读取PDF
    text = load_pdf(pdf_path)
    # 2. 清洗
    text = clean_text(text)
    # 3. 切块
    chunks = split_text(text)
    # 4. 向量化
    embeddings = get_embeddings(chunks)
    # 5. 建立向量库
    index = build_faiss_index(embeddings)

    return chunks, index

def ask_pdf(query, chunks, index, history):
    # rag搜索结果
    results = retrieve(query, model, index, chunks, k=10)

    # 只记录最近3轮问答 6组
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
    # 拼context
    context = "\n\n".join(results)
    prompt = f"""
            You are an academic paper reading assistant.
            请使用中文回答。
            你拥有两种信息来源：
            1. 历史对话
            {history_text}
            2. 当前论文内容
            {context}
            规则：
            - 如果用户询问之前的问题、回答、讨论内容，
            请根据历史对话回答。

            - 如果用户询问论文内容，
            请根据论文内容回答。
            - 如果当前问题涉及到之前的问题，
            请结合论文内容和历史对话回答。

            - 不要编造信息。

            历史对话:
            {history_text}

            Context:
            {context}

            Question:
            {query}
            """
    # 调用LLM
    # answer = llm(prompt)
    
    # return answer

    yield from llm_stream(prompt)