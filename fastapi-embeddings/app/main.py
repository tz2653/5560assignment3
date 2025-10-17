# English comments
from fastapi import FastAPI
from .nlp import EmbeddingRequest, embed_text

app = FastAPI(title="FastAPI Embeddings API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed")
def create_embedding(req: EmbeddingRequest):
    vec = embed_text(req.text)
    return {"embedding": vec, "dim": len(vec)}

