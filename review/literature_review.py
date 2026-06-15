"""根据文献搜索结果 总结"""
from review.search_paper import search_papers
from review.rank_paper import rank_papers
from agents.llm import llm_stream

def literature_review(query, n_search=30, n_select=10):
    # 检索n_search篇文章
    papers = search_papers(query, n_search)
    if len(papers) == 0:
        yield """
                ⚠️ 未检索到文献。

                可能原因：
                - OpenAlex服务暂时不可用
                - 网络连接异常
                - OpenAlex访问频率限制

                请稍后重试。
                """
        return
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
    # 提示词，总结每篇论文
    prompt = f"""
            你是一名专业的科研助理，用检索工具帮助用户搜索和总结学术文献。
            请使用中文回答问题。

            你已经用检索工具搜索到相关论文在Papers中
            根据搜索到的每篇论文，请输出每一篇文章的：
            1. 论文标题
            2. 摘要
            3. DOI

            要求：
            
            - 标题用论文原文输出，不用翻译成中文
            - 摘要总结成中文 尽量保留原意
            - 如果摘要缺失，摘要部分回答：搜索结果未包含摘要
            - 不要编造摘要中未提及的信息
            - 输出格式清晰易读, 开头可为：根据搜索结果 

            Question:
            {query}
            
            Papers:
            {context}
            """
    yield from llm_stream(prompt)