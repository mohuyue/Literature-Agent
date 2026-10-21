"""读取 Zotero 文献库。

两类能力：
1. 全库标题检索（建 FAISS 索引，供 open_zotero_paper 按关键词找一篇）；
2. 分类（collection）读取——按分类名或路径取出该分类下的全部论文，
   支持递归子分类，供「读某个分类里的所有论文做综述」使用。
"""
from rag.embedding import model
from rag.vector_store import build_faiss_index
from zotero.zotero_db import connect_db

# ---------------------------------------------------------------------------
# 条目类型白名单
# ---------------------------------------------------------------------------
# 必须包含 bookSection：大量综述/章节被 Zotero 存为「书籍章节」，
# 旧实现只放行 journalArticle/conferencePaper/book，会把它们静默漏掉
# （实测「缺失值」分类 4 篇里就有 1 篇是 bookSection）。
PAPER_ITEM_TYPES = (
    "journalArticle",
    "conferencePaper",
    "book",
    "bookSection",
    "thesis",
    "report",
    "preprint",
    "manuscript",
    "magazineArticle",
    "newspaperArticle",
    "encyclopediaArticle",
)

# 直接拼进 SQL 的类型清单（内容为代码内常量，不接受外部输入）
PAPER_ITEM_TYPES_SQL = "(" + ", ".join("'%s'" % t for t in PAPER_ITEM_TYPES) + ")"

# 用户在 Zotero 界面里看到的根节点「我的文库」，在数据库里并不是一行 collections，
# 而只是 libraryID 常量。解析路径时要把它剥掉。
ROOT_ALIASES = {
    "我的文库",
    "我的文献库",
    "我的图书馆",
    "my library",
    "mylibrary",
    "library",
    "root",
}


def _title_field_id(cursor):
    """取 Zotero 中 title 字段的 fieldID。"""
    cursor.execute("SELECT fieldID FROM fieldsCombined WHERE fieldName='title'")
    row = cursor.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# ① 全库标题（原有能力，类型白名单已补全）
# ---------------------------------------------------------------------------
def load_zotero_library():
    """读取全库所有论文条目的 itemID 与标题。"""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        title_field_id = _title_field_id(cursor)
        if title_field_id is None:
            return []

        cursor.execute(f"""
        SELECT DISTINCT
            items.itemID,
            itemDataValues.value
        FROM items
        JOIN itemTypes
            ON items.itemTypeID = itemTypes.itemTypeID
        JOIN itemData
            ON items.itemID = itemData.itemID
        JOIN itemDataValues
            ON itemData.valueID = itemDataValues.valueID
        LEFT JOIN deletedItems
            ON deletedItems.itemID = items.itemID
        WHERE itemData.fieldID = {title_field_id}
          AND itemTypes.typeName IN {PAPER_ITEM_TYPES_SQL}
          AND deletedItems.itemID IS NULL
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [{"itemID": item_id, "title": title} for item_id, title in rows]


def build_zotero_index(papers):
    """把标题向量化并建立 FAISS 索引。"""
    titles = [paper["title"] for paper in papers]
    embeddings = model.encode(titles)
    return build_faiss_index(embeddings)


def open_zotero_paper(query, papers, index, top_k=1):
    """按关键词/标题在全库中检索最相关的若干篇。"""
    query_embedding = model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(papers):
            results.append(papers[idx])
    return results


# ---------------------------------------------------------------------------
# ② 分类（collection）
# ---------------------------------------------------------------------------
def _all_collections(conn):
    """返回全部分类，含由父子关系拼出的完整路径，如 '机器学习/缺失值'。"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT collectionID, collectionName, parentCollectionID FROM collections"
    )
    by_id = {
        row[0]: {"collectionID": row[0], "name": row[1], "parentID": row[2]}
        for row in cursor.fetchall()
    }
    for cid, col in by_id.items():
        parts, seen, cur_id = [], set(), cid
        while cur_id is not None and cur_id in by_id and cur_id not in seen:
            seen.add(cur_id)
            parts.append(by_id[cur_id]["name"])
            cur_id = by_id[cur_id]["parentID"]
        col["path"] = "/".join(reversed(parts))
    return list(by_id.values())


def _children_map(collections):
    children = {}
    for col in collections:
        children.setdefault(col["parentID"], []).append(col["collectionID"])
    return children


def _descendant_ids(collections, root_id):
    """含自身在内的全部后代分类 ID（深度优先）。"""
    children = _children_map(collections)
    ids, stack = [], [root_id]
    while stack:
        cur = stack.pop()
        ids.append(cur)
        stack.extend(children.get(cur, []))
    return ids


def _normalize(name_or_path):
    """把用户输入的分类名/路径规范化为路径片段列表。

    容忍：反斜杠分隔、中文全角「＞」、多余的斜杠、以及带/不带根节点「我的文库」。
    注意不要把空格当分隔符——分类名内部允许有空格（如「开题背景/机器学习/透明/抗冲」）。
    """
    text = (name_or_path or "").strip()
    for sep in ("\\", "＞", "〉", "→"):
        text = text.replace(sep, "/")
    while "//" in text:
        text = text.replace("//", "/")

    parts = [p.strip() for p in text.split("/") if p.strip()]
    # 剥掉根别名「我的文库」
    if parts and parts[0].lower() in ROOT_ALIASES:
        parts = parts[1:]
    return parts


def resolve_collection(name_or_path, collections):
    """把分类名或路径解析为唯一分类。

    返回 (collection | None, candidates)。
    - 命中唯一：返回 (该分类, [该分类])
    - 找不到或重名：返回 (None, 候选列表)，由调用方决定是否让用户澄清
    """
    parts = _normalize(name_or_path)
    if not parts:
        # 只给了根节点，返回顶层分类作为候选
        top = [c for c in collections if c["parentID"] is None]
        return None, top

    target = "/".join(parts).lower()

    # ① 完整路径匹配
    exact = [c for c in collections if c["path"].lower() == target]
    if len(exact) == 1:
        return exact[0], exact

    # ② 路径尾部匹配（用户常只写后两级）
    suffix = [c for c in collections if c["path"].lower().endswith("/" + target)]
    pool = suffix or exact
    if len(pool) == 1:
        return pool[0], pool

    # ③ 退化为末级分类名匹配
    leaf = parts[-1].lower()
    by_name = [c for c in collections if c["name"].lower() == leaf]
    if len(by_name) == 1:
        return by_name[0], by_name
    if by_name:
        return None, by_name
    if pool:
        return None, pool

    # ④ 模糊包含
    return None, [c for c in collections if leaf in c["name"].lower()]


def _papers_in_collection(cursor, title_field_id, collection_id):
    """取某分类下『直接包含』的论文条目（不含子分类）。"""
    cursor.execute(f"""
    SELECT DISTINCT
        items.itemID,
        itemDataValues.value,
        itemTypes.typeName
    FROM collectionItems
    JOIN items
        ON items.itemID = collectionItems.itemID
    JOIN itemTypes
        ON items.itemTypeID = itemTypes.itemTypeID
    JOIN itemData
        ON items.itemID = itemData.itemID
     AND itemData.fieldID = ?
    JOIN itemDataValues
        ON itemData.valueID = itemDataValues.valueID
    LEFT JOIN deletedItems
        ON deletedItems.itemID = items.itemID
    WHERE collectionItems.collectionID = ?
      AND itemTypes.typeName IN {PAPER_ITEM_TYPES_SQL}
      AND deletedItems.itemID IS NULL
    ORDER BY itemDataValues.value
    """, (title_field_id, collection_id))

    return [
        {"itemID": row[0], "title": row[1], "itemType": row[2]}
        for row in cursor.fetchall()
    ]


def list_collections(keyword=None, include_empty=False, conn=None):
    """列出分类树，附带（含子分类的）论文总数。

    论文数按 itemID 去重统计——同一篇论文可能同时挂在多个子分类下，
    直接累加各子分类的条数会虚高。keyword 非空时只保留路径含该关键词的分类。
    """
    own_conn = conn is None
    if own_conn:
        conn = connect_db()
    try:
        cursor = conn.cursor()
        title_field_id = _title_field_id(cursor)
        collections = _all_collections(conn)

        # 每个分类自身直接包含的论文 itemID 集合
        direct_ids = {}
        for col in collections:
            direct_ids[col["collectionID"]] = {
                paper["itemID"]
                for paper in _papers_in_collection(cursor, title_field_id, col["collectionID"])
            }

        result = []
        for col in collections:
            ids = _descendant_ids(collections, col["collectionID"])
            merged = set()
            for cid in ids:
                merged |= direct_ids.get(cid, set())
            if not include_empty and not merged:
                continue
            if keyword and keyword.lower() not in col["path"].lower():
                continue
            result.append({
                "collectionID": col["collectionID"],
                "name": col["name"],
                "path": col["path"],
                "direct_papers": len(direct_ids.get(col["collectionID"], set())),
                "total_papers": len(merged),
                "subcollections": len(ids) - 1,
            })
        result.sort(key=lambda c: (-c["total_papers"], c["path"]))
        return result
    finally:
        if own_conn:
            conn.close()


def get_collection_papers(name_or_path, include_subcollections=True, conn=None):
    """按分类名或路径取出其中的论文（默认递归含子分类）。

    返回 dict:
      {"status": "ok"|"not_found"|"ambiguous",
       "collection": {...} | None,
       "papers": [{itemID,title,itemType}],
       "candidates": [路径...]}
    """
    own_conn = conn is None
    if own_conn:
        conn = connect_db()
    try:
        cursor = conn.cursor()
        title_field_id = _title_field_id(cursor)
        collections = _all_collections(conn)

        col, candidates = resolve_collection(name_or_path, collections)
        if col is None:
            return {
                "status": "ambiguous" if candidates else "not_found",
                "collection": None,
                "papers": [],
                "candidates": [c["path"] for c in candidates],
            }

        ids = (
            _descendant_ids(collections, col["collectionID"])
            if include_subcollections
            else [col["collectionID"]]
        )

        seen, papers = set(), []
        for cid in ids:
            for paper in _papers_in_collection(cursor, title_field_id, cid):
                if paper["itemID"] in seen:
                    continue
                seen.add(paper["itemID"])
                papers.append(paper)
        papers.sort(key=lambda p: p["title"].lower())

        return {
            "status": "ok",
            "collection": {
                "collectionID": col["collectionID"],
                "name": col["name"],
                "path": col["path"],
                "subcollections": len(ids) - 1,
            },
            "papers": papers,
            "candidates": [col["path"]],
        }
    finally:
        if own_conn:
            conn.close()
