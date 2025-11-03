import io
import base64
import torch
from torchvision.utils import save_image
from helper_lib.model import Generator
from helper_lib.trainer import train_gan

# Lazy global GAN model
_global_gen_model = None
_global_device = torch.device("cpu")

def _load_or_train_gan():
    """
    Make sure we have a trained generator model.
    """
    global _global_gen_model
    if _global_gen_model is None:
        G = train_gan(epochs=1)
        G.eval()
        _global_gen_model = G
    return _global_gen_model

def generate_samples(num_samples: int = 9):
    """
    Use the trained generator to produce fake MNIST-like digit images.
    """
    device = _global_device
    G = _load_or_train_gan()
    G.to(device)

    z = torch.randn(num_samples, G.latent_dim, device=device)
    with torch.no_grad():
        fake_imgs = G(z)
        fake_imgs = (fake_imgs + 1) / 2.0

    return fake_imgs.cpu()

def image_tensor_to_base64_png(tensor_img):
    """
    Convert tensor image to base64 string.
    """
    buf = io.BytesIO()
    save_image(tensor_img, buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()
    b64_str = base64.b64encode(png_bytes).decode("utf-8")
    return b64_str

def sample_gan_images_as_base64(num_samples: int = 9):
    """
    Public API for FastAPI endpoint.
    """
    imgs = generate_samples(num_samples=num_samples)
    results = []
    for i in range(num_samples):
        b64_png = image_tensor_to_base64_png(imgs[i])
        results.append(b64_png)
    return results
