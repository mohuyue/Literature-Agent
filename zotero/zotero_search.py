"""读取Zotero所有标题，建立FAISS索引"""
import sqlite3
from rag.embedding import model
from rag.vector_store import build_faiss_index
from config import ZOTERO_DB_PATH

# zotero.sqlite地址
DB_PATH = ZOTERO_DB_PATH

def load_zotero_library():
    # 连接数据库，打开zotero.sqlite
    conn = sqlite3.connect(DB_PATH)
    # 创建SQL执行器
    cursor = conn.cursor()

    # 找到title对应字段ID，使用SQL语言
    cursor.execute("""
    SELECT fieldID
    FROM fieldsCombined
    WHERE fieldName='title'
    """)
    # 取出fieldID
    title_field_id = cursor.fetchone()[0]

    # 查询所有标题，只读取真正文献条目
    cursor.execute(f"""
    SELECT
        items.itemID,
        itemDataValues.value
    FROM items
                   
    JOIN itemTypes
        ON items.itemTypeID = itemTypes.itemTypeID
    JOIN itemData
        ON items.itemID=itemData.itemID
    JOIN itemDataValues
        ON itemData.valueID=itemDataValues.valueID
    WHERE itemData.fieldID={title_field_id}
    And itemTypes.typeName IN (
        'journalArticle',
        'conferencePaper',
        'book'
    )
    """)
    # 读取结果
    rows = cursor.fetchall()
    # 关闭数据库
    conn.close()

    # 转换成python字典
    papers = []
    for item_id, title in rows:
        papers.append(
            {
                "itemID": item_id,
                "title": title
            }
        )
    return papers

def build_zotero_index(papers):
    titles = [paper["title"] for paper in papers]
    # 标题向量化
    embeddings = model.encode(titles)
    # 创建FAISS索引
    index = build_faiss_index(embeddings)

    return index

def open_zotero_paper(
    query,
    papers,
    index,
    top_k=1
):
    # 生成query向量
    query_embedding = model.encode([query]).astype("float32")
    # FAISS搜索，计算问题向量与标题向量库中向量的距离
    _, indices = index.search(
        query_embedding,
        top_k
    )
    # 存放检索出的标题
    results = []
    for idx in indices[0]:
        results.append(papers[idx])

    return results