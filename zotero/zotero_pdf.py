"""获取zotero文献搜索结果的pdf地址"""
import sqlite3
import os
from config import ZOTERO_DB_PATH, ZOTERO_STORAGE_PATH

# zotero.sqlite地址
DB_PATH = ZOTERO_DB_PATH
# zotero存储文献的文件夹
STORAGE_PATH = ZOTERO_STORAGE_PATH

def get_pdf_path(paper_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 找附件
    cursor.execute("""
    SELECT 
        itemID, 
        path
    FROM itemAttachments
    WHERE parentItemID = ?
    """, (paper_id,))

    attachment = cursor.fetchone()

    if attachment is None:
        conn.close()
        return None

    attachment_id = attachment[0]
    filename = attachment[1].replace("storage:","")

    # 找storage key
    cursor.execute("""
    SELECT key
    FROM items
    WHERE itemID = ?
    """, (attachment_id,))

    storage_key = cursor.fetchone()[0]

    conn.close()

    pdf_path = os.path.join(
        STORAGE_PATH,
        storage_key,
        filename
    )
    if os.path.exists(pdf_path):
        return pdf_path
    
    return None