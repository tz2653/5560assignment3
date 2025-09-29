from fastapi import APIRouter
from pydantic import BaseModel
import spacy

# 加载 spaCy 模型（中等大小的英文模型）
nlp = spacy.load("en_core_web_md")

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

class TextInput(BaseModel):
    text: str

class TextPairInput(BaseModel):
    text1: str
    text2: str

@router.post("/vector")
def get_vector(input: TextInput):
    doc = nlp(input.text)
    if len(doc) == 0:
        return {"error": "No tokens found"}
    
    vector = doc[0].vector.tolist()
    return {"text": input.text, "vector": vector[:10]}  

@router.post("/similarity")
def get_similarity(input: TextPairInput):
    doc1 = nlp(input.text1)
    doc2 = nlp(input.text2)
    similarity = doc1.similarity(doc2)
    return {"text1": input.text1, "text2": input.text2, "similarity": similarity}
