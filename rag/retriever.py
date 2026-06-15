"""文本检索器"""
import numpy as np

def retrieve(
    query,
    model,
    index,
    chunks,
    k=5
):
    # 问题向量化，和文档向量化用同一个模型
    query_embedding = model.encode([query]).astype("float32") # [query]才能返回二维数组

    # FAISS搜索，计算问题向量与文档中向量的距离
    _, indices = index.search(
        query_embedding,
        k  # 返回与query_embedding最相关的k个Chunk
    )

    # 存放检索出的文本
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results