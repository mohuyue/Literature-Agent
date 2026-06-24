"""建立向量库"""
import faiss
import numpy as np

def build_faiss_index(embeddings):

    # 转 float32
    embeddings = embeddings.astype("float32")

    # 防御：如果 encode 返回了空数组或 1D 数组，提前报清晰错误
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(
            f"向量化结果异常：embeddings.shape={embeddings.shape}。"
            "可能原因：PDF 文本提取为空，或 clean_text 误删了正文内容。"
        )
    
    # 归一化
    faiss.normalize_L2(embeddings)

    # 向量维度
    dimension = embeddings.shape[1] 

    # 创建FAISS索引
    index = faiss.IndexFlatIP(dimension) # 创建一个容器用于存放384维向量

    # 把所有Embedding加入FAISS
    index.add(embeddings) 
    
    return index