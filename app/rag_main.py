import json
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("artifacts/rag_embedder")

# Load FAISS index
faiss_index = faiss.read_index("artifacts/rag_index.faiss")

# Load the document chunks
with open("artifacts/rag_chunks.json", "r", encoding="utf-8") as f:
    text_chunks = json.load(f)

# Load a QA model (you can swap with a better one if needed)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to answer user question
def answer_question(query, top_k=3):
    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Retrieve top-k similar chunks from FAISS index
    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), top_k)

    # Concatenate retrieved chunks
    context = " ".join([text_chunks[i] for i in I[0] if i < len(text_chunks)])

    if not context.strip():
        return "❗ No relevant context found."

    # Get answer from QA model
    result = qa_pipeline(question=query, context=context)
    return result["answer"]
