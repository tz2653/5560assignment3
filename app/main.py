from fastapi import FastAPI
from pydantic import BaseModel

# GAN image generation utilities (Module 6)
from app.gan_api_utils import sample_gan_images_as_base64

# RNN/LSTM text generation utilities (Module 7)
from app.rnn_textgen import generate_with_rnn_inference

# Create FastAPI app
app = FastAPI(
    title="FastAPI GAN + TextGen API",
    version="0.3.0"
)

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Module 6: GAN generation endpoint
@app.get("/gan_generate")
def gan_generate_endpoint(num_samples: int = 9):
    """
    Generate MNIST-like digit images using a trained GAN.
    Returns a list of base64-encoded PNG strings.
    """
    images_b64 = sample_gan_images_as_base64(num_samples=num_samples)
    return {
        "num_samples": num_samples,
        "images": images_b64
    }

# Request model for text generation
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

# Module 7: RNN/LSTM text generation endpoint
@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """
    Generate text from a starting string using a simple character-level LSTM.
    """
    text = generate_with_rnn_inference(
        start_word=request.start_word,
        length=request.length
    )
    return {"generated_text": text}
