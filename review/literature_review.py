"""根据文献搜索结果 总结"""
from review.search_paper import search_papers
from review.rank_paper import rank_papers

def search_online_literature(query, n_search=30, n_select=10):
    # 检索n_search篇文章
    papers = search_papers(query, n_search)
    if len(papers) == 0:
        return """⚠️ 未检索到文献"""
    
    # 根据问题相关性排序后的前n_select篇文章
    top_papers = rank_papers(query, papers, top_k=min(n_select, len(papers)))

    # 拼接文章题目、摘要
    context = ""
    for i, paper in enumerate(top_papers):
        abstract = paper.get(
                        "abstract",
                        "No abstract available.")[:500]
        context += f"""
Paper {i+1}
Title: {paper["title"]}
Abstract: {abstract}
DOI: {paper["doi"]}
==================================
"""
    return context