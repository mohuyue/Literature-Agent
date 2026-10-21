"""获取 zotero 文献条目对应的 PDF 本地路径。"""
import os

from config import ZOTERO_STORAGE_PATH
from zotero.zotero_db import connect_db

# ZOTERO_STORAGE_PATH 形如 <Zotero 数据目录>/storage
_STORAGE_PATH = os.path.normpath(ZOTERO_STORAGE_PATH)
_DATA_DIR = os.path.dirname(_STORAGE_PATH)


def _resolve_attachment_path(conn, attachment_item_id, raw_path):
    """把 itemAttachments.path 解析成磁盘上的绝对路径。

    Zotero 的 path 有三种形式：
      storage:<filename>     → 库内存储，位于 <数据目录>/storage/<附件 key>/<filename>
      attachments:<relpath>  → 链接文件，相对 <数据目录>
      <abs/rel path>         → 链接文件，相对「链接附件基目录」（未配置时按数据目录猜）
    """
    if not raw_path:
        return None

    if raw_path.startswith("storage:"):
        cursor = conn.cursor()
        cursor.execute("SELECT key FROM items WHERE itemID = ?", (attachment_item_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return os.path.join(_STORAGE_PATH, row[0], raw_path[len("storage:"):])

    if raw_path.startswith("attachments:"):
        return os.path.join(_DATA_DIR, raw_path[len("attachments:"):])

    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(_DATA_DIR, raw_path)


def get_pdf_path(paper_id):
    """返回该条目下第一个「存在且非空」的 PDF 附件路径，没有则返回 None。

    与原实现的区别：
    - 优先取 contentType='application/pdf'（原实现只看第一条附件，可能拿到网页快照）；
    - 跳过 path 为空、文件不存在、大小为 0 的附件，而不是直接放弃整条记录。
    """
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT itemID, path, contentType
        FROM itemAttachments
        WHERE parentItemID = ?
        ORDER BY (contentType = 'application/pdf') DESC, itemID
        """, (paper_id,))

        first_resolved = None
        for attachment_item_id, raw_path, content_type in cursor.fetchall():
            full_path = _resolve_attachment_path(conn, attachment_item_id, raw_path)
            if not full_path or not os.path.isfile(full_path):
                continue
            if os.path.getsize(full_path) == 0:
                continue
            is_pdf = (content_type or "").lower() == "application/pdf" \
                or full_path.lower().endswith(".pdf")
            if is_pdf:
                return full_path
            # 非 PDF 先记下，仅当没有任何 PDF 时退回使用
            if first_resolved is None:
                first_resolved = full_path

        return first_resolved
    finally:
        conn.close()
