# Python RAG

> 本專案是學習用的資源，程式碼來自 [**MarkTechStation** 的使用Python构建RAG系统/rag](https://github.com/MarkTechStation/VideoCode/blob/main/使用Python构建RAG系统/rag/main.ipynb)


## RAG 是什麼

在這裡，先來簡單介紹一下什麼是 **RAG**（Retrieval-Augmented Generation，檢索增強生成）。

你有沒有遇過這種情況？明明只是問 LLM 一個問題，但因為資訊太多、太複雜，或是根本不在它的知識範圍內，它卻還是能一本正經地回答你一大串。最可怕的是，它講得越有自信，你反而越容易相信。 結果查證後才發現，根本是在胡說八道。

這就是所謂的 Hallucination（幻覺）。

而 RAG 的想法其實很直覺：既然你不確定答案，那為什麼不先去查資料？

與其讓 LLM 單靠自己的記憶硬掰，不如先從我們準備好的知識庫中找出與問題最相關的內容，再把這些資料交給 LLM，讓它有憑有據地回答。

簡單來說，RAG 就是幫 LLM 養成一個好習慣：不知道就先查，不要裝懂。

當然，RAG 不能完全消除幻覺，但至少能大幅降低模型憑空編造答案的機會。

下面是我簡單整理的一套RAG的基本流程(想要學習更深入的話)：

## 1.資料分片流程

把原始文件切成一段一段的小塊，方便之後各自轉成向量，跟著原文本一起進入資料庫。

```
doc.md ─▶ 分片（Chunking）─▶ 多個片段（Chunk）
```

- **分片（Chunking）**：以空行為單位，把 `doc.md` 切成多個段落
- **片段（Chunk）**：切出來的每一小段文字，是之後檢索的最小單位

## 二、提問 LLM 前流程（建立索引）

在使用者提問之前，先把所有片段轉成向量存起來。

```
片段（Chunk）─▶ 嵌入模型（Embedding Model）─▶ 向量（Vector）─▶ 向量資料庫（Vector Database）
```

- **嵌入模型（Embedding Model）**：把文字轉成一串數字的模型，這裡用 `shibing624/text2vec-base-chinese`
- **向量（Vector / Embedding）**：文字轉出來的數字陣列，意思越接近的文字，向量也越接近
- **正規化（Normalization）**：把向量長度統一成 1，比較相似度時只看方向
- **向量資料庫（Vector Database）**：專門存向量、找相似向量的資料庫，這裡用 ChromaDB（記憶體模式，程式結束就清空）

## 三、提問後流程

使用者提問後，找出相關片段，再交給 LLM 回答。

```
問題（Query）─▶ 嵌入模型（Embedding Model）─▶ 相似度比對（Similarity）─▶ 召回（Retrieve）top 5 ─▶ 重排（Rerank）top 3 ─▶ 生成（Generate）
```

- **相似度比對（Similarity）**：問題也透過嵌入模型轉成向量，再和向量資料庫裡每個片段的向量比對有多接近，常用餘弦相似度（Cosine Similarity）
- **召回（Retrieve / Retrieval）**：依相似度取出最接近的 5 段。速度快但比較粗略
- **重排（Rerank）**：用交叉編碼器（Cross-Encoder）對「問題 + 片段」逐對打分，留下最相關的 3 段。比較慢但更精準
- **交叉編碼器（Cross-Encoder）**：把問題和片段一起丟進模型算分數，這裡用 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **生成（Generate / Generation）**：把問題和這 3 段組成提示詞（Prompt），交給大型語言模型（LLM, Large Language Model）Gemini 回答
- **提示詞（Prompt）**：送給 LLM 的完整輸入文字，包含指令、問題和相關片段

## 完整流程

```
【資料分片】
doc.md ─▶ 分片 ─▶ 片段

【提問 LLM 前】
片段 ─▶ 嵌入模型 ─▶ 向量 ─▶ 向量資料庫
                                │
【提問後】                       ▼
問題 ─▶ 嵌入模型 ─▶ 相似度比對 ─▶ 召回 top 5 ─▶ 重排 top 3 ─▶ LLM 生成回答
```

## 專案結構

| 檔案 | 說明 |
| --- | --- |
| `main.py` | 完整 RAG 流程，可直接執行 |
| `main.ipynb` | 原始 notebook，一步步看每個階段的輸出 |
| `doc.md` | 範例文件（哆啦A夢 × 超級賽亞人的故事） |

## 環境準備

需要先安裝：

- uv：https://docs.astral.sh/uv/getting-started/installation/
- Jupyter（只跑 `main.py` 可以不用）：https://jupyter.org/install

下載專案：

```bash
git clone https://github.com/ylx959/python_rag.git
cd python_rag
```

在專案根目錄建立 `.env`：

```env
GEMINI_API_KEY=xxx
```

`xxx` 換成你的 Gemini API Key，沒有的話到 https://aistudio.google.com/apikey 申請。

安裝依賴：

```bash
uv sync
```

## 執行

注意！！如果要跑 `main.py`：

```bash
uv run main.py
```

或開 Jupyter 跑 notebook：

```bash
uv run --with jupyter jupyter lab
```

> **記得要用 `uv run --with jupyter jupyter lab`** 開 notebook。直接執行 `jupyter lab` 不會用到這個專案的虛擬環境，會找不到依賴。

> 第一次執行會自動從 Hugging Face 下載 embedding 和 CrossEncoder 模型，需要等一下。下載是由這兩行觸發的：
>
> - `main.py:32`（notebook 第 2 個 cell）：`embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")`
> - `main.py:72`（notebook 第 6 個 cell）：`cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')`
>
> 下載後會快取在 `~/.cache/huggingface/hub`，之後執行就不用再下載。
