from rag.pdf_loader import load_pdf
from rag.cleaner import clean_text
from rag.text_splitter import split_text
from rag.embedding import get_embeddings, model
from rag.vector_store import build_faiss_index
from rag.retriever import retrieve

def build_pipeline(pdf_path):

    # 1. 读取PDF
    text = load_pdf(pdf_path)
    # 2. 清洗
    text = clean_text(text)
    # 3. 切块
    chunks = split_text(text)
    # 4. 向量化
    if not chunks:
        raise ValueError(f"文本切块结果为空，请检查 PDF 是否可提取文本：{pdf_path}")
    embeddings = get_embeddings(chunks)
    # 5. 建立向量库
    index = build_faiss_index(embeddings)

    return chunks, index

# 检索与问题相关的论文内容
def retrieve_current_paper(query, chunks, index, history):
    # rag搜索结果
    results = retrieve(query, model, index, chunks, k=5)
    # 只记录最近3轮问答 6组
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
    # 拼context
    context = "\n\n".join(results)

    return f"""
=== Conversation History ===
{history_text}

=== Retrieved Paper Content ===
{context}
"""