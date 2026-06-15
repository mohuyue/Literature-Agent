"""接入大语言模型"""
from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL )

def llm(prompt):
    res = client.chat.completions.create(
        model=LLM_MODEL, # 选择大模型
        messages=[
            {
            "role": "system",
            "content": """
                    You are a professional academic paper reading assistant.
                    You answer questions based only on the provided context.
                    If the answer is not in the context, say you don't know.
                    """
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return res.choices[0].message.content 

def llm_stream(prompt):
    res = client.chat.completions.create(
        model=LLM_MODEL, # 选择大模型
        messages=[
            {
            "role": "system",
            "content": """
                    You are a professional academic paper reading assistant.
                    You answer questions based only on the provided context.
                    If the answer is not in the context, say you don't know.
                    """
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=True
    )
    # 实现流式输出
    for chunk in res:
        try:
            if len(chunk.choices) == 0:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
        except Exception as e:
            print("Stream chunk error:", e)
            continue