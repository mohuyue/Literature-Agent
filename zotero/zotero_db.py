"""Zotero sqlite 连接工具。

Zotero 客户端运行时会独占 zotero.sqlite，普通连接在执行第一条 SQL 时抛出
`database is locked`。这里统一处理：先按可写方式连接并试跑一条查询，一旦被锁
就退回只读模式（`mode=ro&immutable=1`，不加锁），保证 Agent 在任何时候都能读取
已落盘的内容。

代价：只读模式下看不到 Zotero 尚未 checkpoint 的最新改动，刚在 Zotero 里新增的
条目可能需要重启 Zotero 后才可见。因此这里会打印一条提示，避免"静默读旧数据"。
"""
import sqlite3
from pathlib import Path

from config import ZOTERO_DB_PATH


def _readonly_uri(db_path: str) -> str:
    """构造只读且不加锁的 SQLite URI。"""
    try:
        return Path(db_path).as_uri() + "?mode=ro&immutable=1"
    except ValueError:
        # 路径异常（极少见）时退回手写形式
        return "file:{}?mode=ro&immutable=1".format(db_path.replace("\\", "/"))


def connect_db(quiet: bool = False) -> sqlite3.Connection:
    """连接 zotero.sqlite，被 Zotero 锁定则自动回退只读模式。"""
    conn = sqlite3.connect(ZOTERO_DB_PATH)
    try:
        # 必须有真实查询才会触发加锁，PRAGMA 不触发
        conn.execute("SELECT COUNT(*) FROM collections").fetchone()
        return conn
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            conn.close()
            raise
        conn.close()

    if not quiet:
        print(
            "[zotero] 数据库被 Zotero 客户端锁定，已改用只读模式读取；"
            "刚在 Zotero 中新增的条目可能需重启 Zotero 后才可见。",
            flush=True,
        )
    return sqlite3.connect(_readonly_uri(ZOTERO_DB_PATH), uri=True)
