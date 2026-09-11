"""论文 pipeline 结果的磁盘缓存。

缓存的是一次 build_pipeline 的完整产物：chunks 列表 + FAISS 索引。
目的：关掉 Streamlit 再打开同一篇论文时，跳过 PDF 提取与向量化，直接读取。

与 .vision_cache 同思路，区别在于缓存的粒度是「整篇论文」而非「单张图」。

缓存键包含所有会影响产物内容的因素，任一变化即自动失效：
  PDF 绝对路径 | mtime | size        —— 文件被替换/修改
  PIPELINE_VERSION                   —— 切块/清洗/图块逻辑变更
  VISION_ENABLED / VISION_MODEL      —— 是否含图块、图块描述来自哪个模型
  EMBED_MODEL_ID                     —— 换 embedding 模型则旧索引不兼容
  CHUNK_SIZE / CHUNK_OVERLAP         —— 切块参数变更
"""
import os
import json
import hashlib

import faiss

from config import (
    PIPELINE_VERSION,
    VISION_ENABLED,
    VISION_MODEL,
    EMBED_MODEL_ID,
)
from rag.text_splitter import CHUNK_SIZE, CHUNK_OVERLAP

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paper_cache")

# 视觉描述失败时写入的占位符：含任一标记的 chunk 视为不完整，不写入缓存，
# 否则失败结果会被永久冻结（之后重试成功也读不回来）。
_INCOMPLETE_MARKERS = ("视觉描述失败", "未能生成描述")


def _cache_key(pdf_path):
    st = os.stat(pdf_path)
    raw = "|".join([
        os.path.abspath(pdf_path),
        str(st.st_mtime_ns),
        str(st.st_size),
        PIPELINE_VERSION,
        str(VISION_ENABLED),
        str(VISION_MODEL),
        EMBED_MODEL_ID,
        str(CHUNK_SIZE),
        str(CHUNK_OVERLAP),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _is_complete(chunks):
    """chunks 中不含视觉失败占位符才算完整，可安全缓存。"""
    return not any(m in c for c in chunks for m in _INCOMPLETE_MARKERS)


def load_cached_pipeline(pdf_path):
    """命中返回 (chunks, index)，否则返回 None。任何异常都当作未命中，不影响主流程。"""
    try:
        key = _cache_key(pdf_path)
    except OSError:
        return None

    cpath = os.path.join(_CACHE_DIR, key + ".json")
    ipath = os.path.join(_CACHE_DIR, key + ".faiss")
    if not (os.path.exists(cpath) and os.path.exists(ipath)):
        return None

    try:
        with open(cpath, "r", encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]
        index = faiss.read_index(ipath)
        if len(chunks) != index.ntotal:
            # 内容与索引不一致（如写入被中断），视为失效
            print(
                f"[paper_cache] 缓存不一致（chunks={len(chunks)} vs ntotal={index.ntotal}），忽略",
                flush=True,
            )
            return None
        return chunks, index
    except Exception as e:
        print(f"[paper_cache] 读取缓存失败，忽略：{type(e).__name__}: {e}", flush=True)
        return None


def save_cached_pipeline(pdf_path, chunks, index):
    """写入缓存。失败只告警，不影响本次已得到的结果。"""
    if not _is_complete(chunks):
        print("[paper_cache] 检测到视觉描述占位符，本次不写缓存（避免冻结失败结果）", flush=True)
        return

    try:
        key = _cache_key(pdf_path)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cpath = os.path.join(_CACHE_DIR, key + ".json")
        ipath = os.path.join(_CACHE_DIR, key + ".faiss")

        # 先写临时文件再原子替换，避免写入中断留下半成品缓存
        tmp_c, tmp_i = cpath + ".tmp", ipath + ".tmp"
        with open(tmp_c, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False)
        faiss.write_index(index, tmp_i)
        os.replace(tmp_c, cpath)
        os.replace(tmp_i, ipath)

        print(f"[paper_cache] 已写入缓存：{len(chunks)} chunks", flush=True)
    except Exception as e:
        print(f"[paper_cache] 写缓存失败（不影响本次结果）：{type(e).__name__}: {e}", flush=True)
