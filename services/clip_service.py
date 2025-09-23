# services/clip_service.py
import os, torch, numpy as np, open_clip
CLIP_MODEL_NAME = os.getenv("EMBED_MODEL", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("EMBED_PRETRAINED", "laion2b_s34b_b79k")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = _tok = None

def _load():
    global _model, _tok
    if _model is None:
        _model, _, _ = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
        )
        _model.eval()
        _tok = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return _model, _tok

@torch.inference_mode()
def embed_text_clip_512(text: str) -> list[float]:
    t = (text or "").strip()
    if not t: return [0.0]*512
    m, tok = _load()
    toks = tok([t]).to(DEVICE)
    z = m.encode_text(toks)
    z = z / z.norm(dim=-1, keepdim=True)
    return z[0].float().cpu().numpy().tolist()

