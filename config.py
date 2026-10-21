from dotenv import load_dotenv
import os

load_dotenv()

# LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")

# Vision (图片描述) —— 是否在打开论文时用多模态模型描述图片
# 说明：该步骤需联网逐张调用视觉模型，SiliconFlow 对并发有限流，实测 9 张图约 60s+ 且常超时。
#       默认关闭，保证「打开论文」在数秒内完成；需要读图时再显式开启。
VISION_ENABLED = os.getenv("VISION_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "30"))
VISION_MAX_WORKERS = int(os.getenv("VISION_MAX_WORKERS", "4"))
# 视觉描述的输出上限。推理型视觉模型（如 deepseek-flash）会把 token 先用于 reasoning，
# 上限过小会导致 message.content 为空、只剩思考文本，必须留足。
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "3000"))
# 关闭「思考」：推理型视觉模型（deepseek-flash）默认先长篇 reasoning，
# 既慢又常把 content 挤空。设为 true 时传 reasoning_effort="none"，实测 1.5s/张且输出干净。
VISION_DISABLE_THINKING = os.getenv("VISION_DISABLE_THINKING", "true").strip().lower() in ("1", "true", "yes", "on")

# zotero Configuration
ZOTERO_DB_PATH = os.getenv("ZOTERO_DB_PATH")
ZOTERO_STORAGE_PATH = os.getenv("ZOTERO_STORAGE_PATH")

# ===== Zotero 分类（collection）批量读取 =====
# 单次批量装载的篇数上限：超过时不直接加载，而是先把「该分类及其子分类」的清单
# 返回给用户确认（避免误点「机器学习」这类大分类，一次塞进几十篇导致长时间等待）。
ZOTERO_MAX_COLLECTION_PAPERS = int(os.getenv("ZOTERO_MAX_COLLECTION_PAPERS", "10"))
# 跨论文检索时「每篇」取回的片段数。做综述时必须保证每篇都被覆盖到，
# 所以按篇配额而不是全局 top-k（全局 top-k 会让某一篇占满结果）。
RETRIEVE_PER_PAPER_K = int(os.getenv("RETRIEVE_PER_PAPER_K", "4"))

# search tool Configuration
Paper_Search_BASE_URL = os.getenv("Literature_Search_BASE_URL")

# ===== 论文 pipeline 磁盘缓存（rag/paper_cache.py）=====
# 缓存 chunks + FAISS 索引，避免每次打开论文都重新提取与向量化。
# PIPELINE_VERSION：任何影响 chunk 内容或索引的改动（切块参数、清洗逻辑、图块标签格式等）
#                   都必须 +1，否则会静默读到过期缓存。
PIPELINE_VERSION = "1"
# embedding 模型标识：换模型 → 向量维度可能变化 → 旧索引不可用，故纳入缓存键。
EMBED_MODEL_ID = "AI-ModelScope/bge-small-en-v1.5"