from dotenv import load_dotenv
import os

load_dotenv()

# LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

# zotero Configuration
ZOTERO_DB_PATH = os.getenv("ZOTERO_DB_PATH")
ZOTERO_STORAGE_PATH = os.getenv("ZOTERO_STORAGE_PATH")

# search tool Configuration
Paper_Searcg_BASE_URL = os.getenv("Literature_Search_BASE_URL")