"""文献搜索 OpenAlex API检索"""
import requests
from config import Paper_Search_BASE_URL
# 定义恢复摘要的函数
def reconstruct_abstract(inverted_index):

    if inverted_index is None:
        return ""
    
    words = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))

    words.sort()
    return " ".join(word for pos, word in words)

# 定义搜索函数
def search_papers(query, n_search):
    
    url = (Paper_Search_BASE_URL)  
    params = {
        "search": query,
        "per-page": n_search # 搜索的论文数量
    }
    # 搜索
    try:
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 200:
            raise Exception(f"OpenAlex返回状态码: {r.status_code}")

        data = r.json()
        papers = data.get("results", [])

    except Exception as e:
        print("OpenAlex搜索失败")
        print(e)
        return []

    results = []

    # 提取搜索结果中的有用信息
    results = []
    for paper in papers:
        results.append({
            "title": paper.get("title", ""),
            "year": paper.get("publication_year"),
            "doi": paper.get("doi", ""),
            "citation": paper.get("cited_by_count", 0),
            "abstract":reconstruct_abstract(
                        paper.get("abstract_inverted_index"))
        })

    return results