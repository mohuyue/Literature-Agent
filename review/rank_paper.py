""" 根据与问题的相似度对论文进行重排序 """
from sklearn.metrics.pairwise import cosine_similarity
from rag.embedding import model

def rank_papers(
    query,
    papers,
    top_k=5
):
    # 论文文本
    paper_texts = []

    for paper in papers:
        title = paper.get("title", "")

        abstract = paper.get("abstract", "")

        paper_texts.append(
            f"{title}\n{abstract}"
        )

    # 论文Embedding
    paper_embeddings = model.encode(paper_texts)

    # Query Embedding
    query_embedding = model.encode([query])

    # 相似度
    scores = cosine_similarity(
        query_embedding,
        paper_embeddings
    )[0]

    # 保存分数
    for paper, score in zip(papers, scores):
        paper["score"] = float(score)

    # 排序
    ranked_papers = sorted(
        papers,
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked_papers[:top_k]