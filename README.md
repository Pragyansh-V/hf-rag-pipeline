
# HuggingFace RAG Pipeline

End-to-end Retrieval Augmented Generation (RAG) pipeline built from 
scratch using HuggingFace, FAISS, and Groq's Llama 3.3 70B. Answers 
questions grounded in a document knowledge base — with citations, 
no hallucination on out-of-scope questions.

---

## What this project covers

- Document chunking with configurable size and overlap
- Semantic embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
- Vector similarity analysis — cosine similarity matrix across documents
- FAISS vector index (IndexFlatL2) for fast nearest-neighbour retrieval
- Grounded generation via Groq API (Llama 3.3 70B) with context injection
- Hallucination prevention — model says "I don't know" when answer 
  is not in documents
- Gradio chat interface with source citations

---

## How it works

Question
→ Embed with sentence-transformers
→ Search FAISS index (L2 distance)
→ Retrieve top-3 relevant chunks
→ Inject chunks into Groq/Llama prompt
→ Generate grounded answer with citations
→ Display in Gradio chat interface

---

## Key findings

**Retrieval works semantically, not by keyword matching**
Query: "How do I use a model without downloading it?"
Retrieved: Inference API doc (distance 1.26) — correct, despite 
sharing no exact keywords with the query.

**Hallucination prevention works**
Query: "What is the price of HuggingFace Pro subscription?"
Answer: "I don't have enough information to answer that."
The model correctly refused to invent an answer not in the documents.

**Semantic similarity reveals conceptual relationships**
Hub and Inference API scored 0.68 cosine similarity — highest pair —
because both describe ways to access hosted models, even though 
they discuss different topics.

---

## Example questions and answers

**Q: What is LoRA and how does it reduce parameters?**
A: LoRA works by adding low-rank matrices to existing weights, reducing 
trainable parameters by up to 99% while maintaining performance.
(Source: What is Fine-tuning?)

**Q: What paper introduced the transformer architecture?**
A: The paper 'Attention is All You Need' introduced transformers in 2017.
(Source: What are Transformers?)

**Q: What is the price of HuggingFace Pro?**
A: I don't have enough information to answer that.

---

## Stack

Python 3.14 · sentence-transformers 5.5.1 · faiss-cpu 1.14.2 · 
langchain 1.3.2 · groq 1.2.0 · gradio 6.15.1 · HuggingFace datasets

---

## How to run

```bash
git clone https://github.com/Pragyansh-V/hf-rag-pipeline.git
cd hf-rag-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your-key-here" >> .env

jupyter notebook rag.ipynb
```

---

## Architecture decisions

**Why FAISS over Pinecone/Weaviate?**
FAISS runs entirely in memory — no external service, no API key, 
no infrastructure. For learning and prototypes it's the right tool. 
For production with millions of documents, swap to Pinecone or 
Weaviate with the same retrieval logic.

**Why all-MiniLM-L6-v2?**
Fast, lightweight (22M parameters), runs on CPU, and produces 
384-dimensional embeddings that capture semantic meaning well for 
English text. Good balance of speed and quality for RAG prototypes.

**Why Groq over OpenAI/Anthropic?**
Free tier with generous limits, ~800 tokens/second inference speed, 
and Llama 3.3 70B is a strong open-source model. No credit card needed.

---

## Limitations

- Knowledge base is small (6 documents) — production RAG needs 
  hundreds or thousands of documents
- No persistent storage — index rebuilds on every notebook run
- No re-ranking step — production RAG uses a second model to 
  re-rank retrieved chunks by relevance
- English only — multilingual RAG requires a multilingual embedding model

---

## Future improvements

- Add persistent FAISS index saved to disk
- Implement chunk re-ranking with a cross-encoder model
- Expand knowledge base with real HuggingFace documentation
- Add metadata filtering — retrieve only from specific document categories
- Evaluate retrieval quality with RAGAS framework

---

## Part of a learning series

- Project 1: [SST-2 Sentiment Model Audit](https://github.com/Pragyansh-V/hf-sentiment-audit)
- Project 2: [DistilBERT Emotion Classifier with LoRA](https://github.com/Pragyansh-V/-hf-finetuning-distilbert)
- Project 3: HuggingFace RAG Pipeline ← you are here
- Project 4: Coming soon — Bias audit & evaluation