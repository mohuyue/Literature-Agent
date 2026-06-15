"""建立向量库"""
import faiss
import numpy as np

def build_faiss_index(embeddings):

    # 转 float32
    embeddings = embeddings.astype("float32")

    # 归一化
    faiss.normalize_L2(embeddings)

    # 向量维度
    dimension = embeddings.shape[1] 

    # 创建FAISS索引
    index = faiss.IndexFlatIP(dimension) # 创建一个容器用于存放384维向量

    # 把所有Embedding加入FAISS
    index.add(embeddings) 
    
    return index