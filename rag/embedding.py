"""向量化"""
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

# 第一次会下载模型
model_dir = snapshot_download("AI-ModelScope/bge-small-en-v1.5")
# 加载模型 利用一个预训练Embedding模型把文本转换成向量
model = SentenceTransformer(model_dir)

def get_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    return embeddings