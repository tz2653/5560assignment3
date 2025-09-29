from fastapi import FastAPI
from pydantic import BaseModel
from .bigram_model import BigramModel   # 注意：相对导入（因为 app 是包）

app = FastAPI()

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "This is another example sentence.",
    "We are generating text based on bigram probabilities.",
    "Bigram models are simple but effective.",
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}
