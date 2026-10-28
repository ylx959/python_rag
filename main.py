"""
簡易 RAG（Retrieval-Augmented Generation）流程：
1. 切分文件成 chunks
2. 用 embedding 模型將 chunks 向量化
3. 存入 ChromaDB 向量資料庫
4. 根據問題召回（retrieve）相關 chunks
5. 用 CrossEncoder 重新排序（rerank）
6. 交給 Gemini 生成最終回答
"""

from typing import List

import chromadb
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder, SentenceTransformer


# ---------- 1. 文件切分 ----------

def split_into_chunks(doc_file: str) -> List[str]:
    """讀取文件，以空行（段落）為單位切分成多個 chunk。"""
    with open(doc_file, 'r') as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


# ---------- 2. 向量化 ----------

# 中文 embedding 模型，首次執行會自動下載
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")

def embed_chunk(chunk: str) -> List[float]:
    """將文字轉成向量；normalize 後可直接用內積/餘弦相似度比較。"""
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


# ---------- 3. 存入向量資料庫 ----------

# EphemeralClient 只存在記憶體中，程式結束即消失
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    """將每個 chunk 與其向量一起寫入 ChromaDB，id 用索引編號。"""
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


# ---------- 4. 召回 ----------

def retrieve(query: str, top_k: int) -> List[str]:
    """將問題向量化後，從資料庫取出最相似的 top_k 個 chunk。"""
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


# ---------- 5. 重排序 ----------

def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    """用 CrossEncoder 對 (問題, chunk) 逐對打分，比向量相似度更精準，取前 top_k 個。"""
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    # 依分數由高到低排序
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


# ---------- 6. 生成回答 ----------

# 從 .env 讀取 GEMINI_API_KEY / GOOGLE_API_KEY
load_dotenv()
google_client = genai.Client()

def generate(query: str, chunks: List[str]) -> str:
    """把問題與相關片段組成 prompt，交給 Gemini 生成回答。"""
    prompt = f"""你是一位知識助手，請根據使用者的問題和下列片段生成準確的回答。

使用者問題: {query}

相關片段:
{"\n\n".join(chunks)}

請基於上述內容作答，不要編造資訊。"""

    print(f"{prompt}\n\n---\n")

    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def main() -> None:
    # 建立索引：切分 -> 向量化 -> 存入資料庫
    chunks = split_into_chunks("doc.md")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    save_embeddings(chunks, embeddings)

    query = "哆啦A夢使用的3個秘密道具分別是什麼？"

    # 先粗略召回 5 個，再精排保留 3 個
    retrieved_chunks = retrieve(query, 5)
    reranked_chunks = rerank(query, retrieved_chunks, 3)

    answer = generate(query, reranked_chunks)
    print(answer)


if __name__ == "__main__":
    main()
