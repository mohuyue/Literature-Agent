# check_pkgs.py
import importlib.metadata as md
import sys

# 包名 -> import 名 的映射（有些包安装名和导入名不一样）
IMPORT_NAME_MAP = {
    "faiss-cpu": "faiss",
    "python-dotenv": "dotenv",
    "sentence-transformers": "sentence_transformers",
    "pymupdf": "fitz",
    "scikit-learn": "sklearn",
    "langchain-openai": "langchain_openai",
    "langchain-core": "langchain_core",
    "langchain-community": "langchain_community",
}

with open("requirements.txt", encoding="utf-8") as f:
    pkgs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

missing, ok = [], []
for pkg in pkgs:
    try:
        ver = md.version(pkg)
        ok.append((pkg, ver))
    except md.PackageNotFoundError:
        missing.append(pkg)

print("=" * 50)
print(f"✅ 已安装 ({len(ok)}/{len(pkgs)}):")
for p, v in ok:
    print(f"   {p:<25} {v}")

if missing:
    print("\n❌ 缺失:")
    for p in missing:
        print(f"   {p}")
    sys.exit(1)
else:
    print("\n🎉 全部已安装")