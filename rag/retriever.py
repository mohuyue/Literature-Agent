"""文本检索器"""
import re
import numpy as np

# 从问题里提取图号：图5 / 图 5 / Figure 5 / Fig. 5
_FIG_IN_QUERY = re.compile(r'(?:图|Figure|Fig\.?)\s*(\d+)', re.I)

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

    # 图号精确命中：向量检索对「图5」这类编号不敏感，
    # 直接把写明该图号的 chunk 提到最前，保证「解读图5」能拿到 Figure 5 的图块。
    fig_nos = set(_FIG_IN_QUERY.findall(query))
    if fig_nos:
        labeled, mentions = [], []   # 图块本体（【Figure 5 ...】）优先于正文里的 "Fig. 5" 引用
        for c in chunks:
            for n in fig_nos:
                if re.search(rf'(?:Figure|Fig\.?|图)\s*{n}\b', c, re.I):
                    (labeled if c.startswith("【Figure") else mentions).append(c)
                    break
        merged = []
        for c in labeled + mentions + results:
            if c not in merged:
                merged.append(c)
        # 命中块置顶，并限制总量避免上下文过长
        results = merged[: k + 3]

    return results