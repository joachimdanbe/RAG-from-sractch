# 🏢 HR Policy Chatbot - RAG Pipeline

A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers questions about company policies using only the provided knowledge base — no hallucinations.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Gradio](https://img.shields.io/badge/Interface-Gradio-orange)

---

## How it works

```
Document → Chunks → Embeddings (MiniLM) → FAISS index
                                                ↓
Question → Embedding → similarity search → top-2 chunks
                                                ↓
                          DistilBERT QA → extracted answer
```

Instead of generating text (which causes hallucination), this pipeline **extracts** the answer directly from the most relevant chunk. If the confidence score is below 0.3, it returns `"I don't have that information."` — no guessing.

---

## Stack

| Component | Model / Library |
|---|---|
| Document splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector search | FAISS `IndexFlatL2` |
| Answer extraction | `distilbert-base-cased-distilled-squad` |
| Interface | Gradio `ChatInterface` |

---

## Installation

```bash
pip install transformers sentence-transformers faiss-cpu langchain langchain-text-splitters gradio
```

---

## Usage

Run all cells in `notebook.ipynb` in order:

1. **Data** — define your knowledge base text
2. **Chunking** — split into 150-char chunks with 20-char overlap
3. **Embeddings + FAISS** — encode chunks and build the index
4. **QA pipeline** — load DistilBERT once globally
5. **Gradio interface** — launch the chatbot

```python
demo.launch()  # opens a public Gradio link
```

---

## Example queries

| Question | Answer |
|---|---|
| What is the WFH policy? | All employees are eligible for a hybrid WFH schedule |
| How many PTO days do employees get? | 20 |
| What is the tech stack? | The official backend language is Python |
| What is the dental plan? | I don't have that information. |

