# fastapi-embeddings

FastAPI service that returns spaCy embeddings for a given word.

## Local (optional)
```bash
uvicorn app.main:app --reload
# then: curl "http://127.0.0.1:8000/api/embed?word=apple"
