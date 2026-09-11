"""文本切块"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 切块参数：纳入 paper_cache 的缓存键，改动后缓存自动失效
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    
    return chunks