# English comments
from functools import lru_cache
import spacy
from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    text: str

@lru_cache(maxsize=1)
def get_nlp():
    # Must match the installed package name from pyproject
    return spacy.load("en_core_web_sm")

def embed_text(text: str) -> list[float]:
    nlp = get_nlp()
    doc = nlp(text)
    return doc.vector.tolist()
