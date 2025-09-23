
import os
import re
import unicodedata
import numpy as np

# ---- VN normalize (match your SQL vn_norm_simple intent) ----
def vn_norm(s: str) -> str:
    s = (s or "").strip().lower()
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # keep a-z 0-9 and separators
    s = re.sub(r"[^a-z0-9 \-\.x/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- Embedding loader: OpenCLIP first, then fallback to OpenAI CLIP ----
# Config via env:
#   EMBED_MODEL=ViT-B-32
#   EMBED_PRETRAINED=laion2b_s34b_b79k
_EMBED_MODEL = os.getenv("EMBED_MODEL", "ViT-B-32")
_EMBED_PRETRAINED = os.getenv("EMBED_PRETRAINED", "laion2b_s34b_b79k")

_model_cache = {"loaded": False, "is_openclip": True, "model": None, "preprocess": None, "device": "cpu"}

def _load_model():
    if _model_cache["loaded"]:
        return _model_cache

    device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else "cpu"

    # Try open_clip first
    try:
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            _EMBED_MODEL, pretrained=_EMBED_PRETRAINED, device=device
        )
        tokenizer = open_clip.get_tokenizer(_EMBED_MODEL)
        _model_cache.update(dict(loaded=True, is_openclip=True, model=model, preprocess=preprocess,
                                 tokenizer=tokenizer, device=device))
        return _model_cache
    except Exception as e:
        # Fallback to OpenAI CLIP
        try:
            import torch
            import clip as openai_clip
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            _model_cache.update(dict(loaded=True, is_openclip=False, model=model, preprocess=preprocess,
                                     tokenizer=None, device=device))
            return _model_cache
        except Exception as e2:
            raise RuntimeError(f"Cannot load any CLIP model: open_clip error={e}; openai-clip error={e2}")

def encode_texts(texts):
    cfg = _load_model()
    import torch
    model = cfg["model"]
    device = cfg["device"]
    with torch.no_grad():
        if cfg["is_openclip"]:
            toks = cfg["tokenizer"](texts)
            if isinstance(toks, dict):  # some tokenizers return dict
                for k in toks: toks[k] = toks[k].to(device)
            else:
                toks = toks.to(device)
            feats = model.encode_text(toks)
        else:
            import clip as openai_clip
            toks = openai_clip.tokenize(texts, truncate=True).to(device)
            feats = model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

def encode_images(pils):
    cfg = _load_model()
    import torch
    model = cfg["model"]
    device = cfg["device"]
    pre = cfg["preprocess"]
    import PIL.Image
    with torch.no_grad():
        imgs = [pre(im.convert("RGB")) for im in pils]
        batch = torch.stack(imgs).to(device)
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

