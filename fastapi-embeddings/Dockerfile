# Use the official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file into the container
COPY requirements.txt .

# Install dependencies and download the spaCy model
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_md

# Copy the entire project into the container
COPY . .

# Run FastAPI with Uvicorn on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
