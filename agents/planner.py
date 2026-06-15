"""规划，将用户问题划分不同任务"""
from agents.llm import llm
import json

def plan(user_query):

    prompt = f"""
        You are an academic literature assistant.
        Determine the user's intent and generate an optimized English query.

        Task types:

        ASK:
        The user is asking about the currently loaded paper.
        Convert the question into natural academic English while preserving the original meaning.

        OPEN_ZOTERO
        The user wants to search or read papers from the Zotero library.
        Extract concise English academic keywords suitable for literature search.
        

        REVIEW:
        The user is requesting related papers, literature reviews,
        research progress, or recent publications.
        Extract concise English academic keywords suitable for literature search.

        Return ONLY valid JSON.

        Examples:

        User:
        总结这篇论文
        Output:
        {{
            "task":"ASK",
            "query":"Summarize this paper."
        }}

        User:
        从Zotero里阅读这篇论文Structure and properties of polypropylene alloy
        Output:
        {{
            "task":"OPEN_ZOTERO",
            "keyword":"Structure and properties of polypropylene alloy"
        }}

        User:
        帮我查找聚丙烯合金相关论文
        Output:
        {{
            "task":"REVIEW",
            "keyword":"polypropylene alloy"
        }}

        User Question:
        {user_query}
        """

    result = llm(prompt)
    return json.loads(result)