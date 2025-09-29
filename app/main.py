from fastapi import FastAPI
from app.routers import embeddings

app = FastAPI(title="FastAPI Word Embeddings API")

app.include_router(embeddings.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Word Embeddings API!"}
