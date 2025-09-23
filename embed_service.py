import os
import numpy as np
import torch
import open_clip
from PIL import Image

MODEL_NAME = os.getenv("EMBED_MODEL", "ViT-B-32")
PRETRAINED = os.getenv("EMBED_PRETRAINED", "laion2b_s34b_b79k")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None
_tokenizer = None

def _lazy_init():
    global _model, _preprocess, _tokenizer
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return (vec / n) if n > 0 else vec

@torch.no_grad()
def embed_image_file(path: str) -> list[float]:
    _lazy_init()
    img = Image.open(path).convert("RGB")
    img = _preprocess(img).unsqueeze(0).to(DEVICE)
    feats = _model.encode_image(img).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().tolist()

@torch.no_grad()
def embed_text(text: str) -> list[float]:
    _lazy_init()
    tokens = _tokenizer([text]).to(DEVICE)
    feats = _model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().numpy().tolist()

