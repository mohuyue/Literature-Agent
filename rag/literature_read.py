from rag.pdf_loader import load_pdf, extract_tables_as_markdown, extract_figures_from_pdf, describe_figures_with_vision
from rag.cleaner import clean_text
from rag.text_splitter import split_text
from rag.embedding import get_embeddings, model
from rag.vector_store import build_faiss_index
from rag.retriever import retrieve
from rag.paper_cache import load_cached_pipeline, save_cached_pipeline
from config import VISION_ENABLED
import time

def build_pipeline(pdf_path):
    print(f"[pipeline] 开始处理：{pdf_path}", flush=True)

    # 0. 先查磁盘缓存：命中则跳过全部提取与向量化（跨重启同样有效）
    cached = load_cached_pipeline(pdf_path)
    if cached is not None:
        chunks, index = cached
        print(f"[pipeline] 命中论文缓存：{len(chunks)} chunks（跳过提取与向量化）", flush=True)
        return chunks, index

    # 1. 读取PDF纯文本
    t = time.time()
    text = load_pdf(pdf_path)
    text = clean_text(text)  # 清洗
    chunks = split_text(text)  # 切块
    print(f"[pipeline] 文本切块完成：{len(chunks)} chunks（{time.time()-t:.1f}s）", flush=True)

    # 2. 提取表格，转为 Markdown
    t = time.time()
    tables_md = extract_tables_as_markdown(pdf_path)
    # 每个 Markdown 表格作为独立 chunk，带来源标记
    table_chunks = []
    for t_md in tables_md:
        table_chunks.append(t_md)
    print(f"[pipeline] 表格提取完成：{len(table_chunks)} tables（{time.time()-t:.1f}s）", flush=True)

    # 3. 提取图（按图注定位）+ 视觉描述
    t = time.time()
    figures = extract_figures_from_pdf(pdf_path)
    print(f"[pipeline] 图提取完成：{len(figures)} figures（{time.time()-t:.1f}s）", flush=True)

    figure_chunks = []
    if VISION_ENABLED and figures:
        # 需联网逐张调用视觉模型；带磁盘缓存，重复打开同一论文时秒级完成
        figures = describe_figures_with_vision(figures, pdf_path=pdf_path)
        for fig in figures:
            desc = fig.get("description")
            if not desc:
                continue
            # 带上真实图号与中文「图N」，保证「解读图5」这类问题能检索到对应图
            figure_chunks.append(
                f"【Figure {fig['figure_no']} (Page {fig['page']}) / 图{fig['figure_no']}】\n"
                f"Caption: {fig['caption']}\n"
                f"图像内容：{desc}"
            )
        print(f"[pipeline] 图描述完成：{len(figure_chunks)}/{len(figures)}（{time.time()-t:.1f}s）", flush=True)
    else:
        print(
            "[pipeline] 已跳过图的视觉描述（VISION_ENABLED=false 或无图）。"
            "如需读图，请在 .env 设置 VISION_ENABLED=true 后重启。",
            flush=True,
        )

    # 4. 合并所有 chunks
    all_chunks = chunks + table_chunks + figure_chunks
    if not all_chunks:
        raise ValueError(f"文本+表格+图片 提取结果均为空，请检查 PDF：{pdf_path}")
    
    # 5. 向量化+建立向量库
    t = time.time()
    embeddings = get_embeddings(all_chunks)
    index = build_faiss_index(embeddings)
    print(f"[pipeline] 向量化+索引完成：{len(all_chunks)} chunks（{time.time()-t:.1f}s）", flush=True)

    # 6. 写入磁盘缓存（含视觉失败占位符时自动跳过），供下次打开直接复用
    save_cached_pipeline(pdf_path, all_chunks, index)

    return all_chunks, index

# 检索与问题相关的论文内容
def retrieve_current_paper(query, chunks, index, history):
    # rag搜索结果
    results = retrieve(query, model, index, chunks, k=3)
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