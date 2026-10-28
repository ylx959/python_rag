# Python RAG

> This project is a learning resource. The code is based on [**MarkTechStation** - 使用Python构建RAG系统/rag (Building a RAG System with Python)](https://github.com/MarkTechStation/VideoCode/blob/main/使用Python构建RAG系统/rag/main.ipynb)


## What is RAG

Let's start with a quick introduction to **RAG** (Retrieval-Augmented Generation).

Have you ever run into this? You ask an LLM a simple question, and even though the information is too much, too complex, or simply outside what it knows, it still gives you a long, serious-sounding answer. The scary part is that the more confident it sounds, the more likely you are to believe it. Then you fact-check it and find out it was just making things up.

This is what's called **Hallucination**.

The idea behind RAG is pretty intuitive: if you're not sure about the answer, why not look it up first?

Instead of letting the LLM bluff from memory alone, we first find the content most relevant to the question from a knowledge base we've prepared, then hand that to the LLM so it can answer based on real evidence.

In short, RAG teaches the LLM a good habit: if you don't know, look it up — don't pretend.

Of course, RAG can't eliminate hallucinations completely, but it greatly reduces the chance of the model inventing answers out of thin air.

Below is a simple overview of the basic RAG pipeline I put together (check the original source if you want to dig deeper):

## 1. Data Chunking

Split the original document into small pieces, so each one can be turned into a vector and stored in the database together with its original text.

```
doc.md ─▶ Chunking ─▶ Chunks
```

- **Chunking**: Split `doc.md` into paragraphs using blank lines as separators
- **Chunk**: Each small piece of text after splitting; the smallest unit for retrieval later

## 2. Before Querying the LLM (Indexing)

Before the user asks anything, convert all chunks into vectors and store them.

```
Chunk ─▶ Embedding Model ─▶ Vector ─▶ Vector Database
```

- **Embedding Model**: A model that turns text into a list of numbers; here we use `shibing624/text2vec-base-chinese`
- **Vector / Embedding**: The array of numbers produced from text; the closer two texts are in meaning, the closer their vectors
- **Normalization**: Scale every vector to length 1, so similarity only depends on direction
- **Vector Database**: A database built to store vectors and find similar ones; here we use ChromaDB (in-memory mode, cleared when the program exits)

## 3. After the Query

After the user asks a question, find the relevant chunks and pass them to the LLM to answer.

```
Query ─▶ Embedding Model ─▶ Similarity ─▶ Retrieve top 5 ─▶ Rerank top 3 ─▶ Generate
```

- **Similarity**: The query is also converted into a vector by the embedding model, then compared against every chunk vector in the database to see how close they are, usually with Cosine Similarity
- **Retrieve / Retrieval**: Take the 5 closest chunks by similarity. Fast but rough
- **Rerank**: Use a Cross-Encoder to score each "query + chunk" pair and keep the 3 most relevant. Slower but more accurate
- **Cross-Encoder**: Feeds the query and a chunk into the model together to compute a score; here we use `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Generate / Generation**: Combine the query and these 3 chunks into a Prompt, and let the LLM (Large Language Model) Gemini answer
- **Prompt**: The full input text sent to the LLM, including instructions, the query, and the relevant chunks

## Full Pipeline

```
[Data Chunking]
doc.md ─▶ Chunking ─▶ Chunks

[Before Querying the LLM]
Chunks ─▶ Embedding Model ─▶ Vectors ─▶ Vector Database
                                              │
[After the Query]                             ▼
Query ─▶ Embedding Model ─▶ Similarity ─▶ Retrieve top 5 ─▶ Rerank top 3 ─▶ LLM generates answer
```

## Project Structure

| File | Description |
| --- | --- |
| `main.py` | The full RAG pipeline, runnable directly |
| `main.ipynb` | The original notebook; see each stage's output step by step |
| `doc.md` | Sample document (an original story, "The Clockmaker's Apprentice of Fog Harbor") |

## Setup

Install first:

- uv: https://docs.astral.sh/uv/getting-started/installation/
- Jupyter (not needed if you only run `main.py`): https://jupyter.org/install

Clone the project:

```bash
git clone https://github.com/ylx959/python_rag.git
cd python_rag
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=xxx
```

Replace `xxx` with your Gemini API key. If you don't have one, get it at https://aistudio.google.com/apikey.

Install dependencies:

```bash
uv sync
```

## Run

Run `main.py`:

```bash
uv run main.py
```

Or open Jupyter to run the notebook:

```bash
uv run --with jupyter jupyter lab
```

> **Remember to use `uv run --with jupyter jupyter lab`** to open the notebook. Launching `jupyter lab` directly won't use this project's virtual environment, so the dependencies won't be found.

> The first run will automatically download the embedding and CrossEncoder models from Hugging Face, so it may take a while. The downloads are triggered by these two lines:
>
> - `main.py:32` (2nd cell in the notebook): `embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")`
> - `main.py:72` (6th cell in the notebook): `cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')`
>
> The models are cached in `~/.cache/huggingface/hub`, so later runs won't download them again.
