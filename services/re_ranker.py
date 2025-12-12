"""Lightweight re-ranker service with safe feature handling."""

from typing import List, Dict
import os
import json
import logging

_HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
    import numpy as np
except Exception:
    _HAS_TORCH = False


class SimpleRanker(nn.Module if _HAS_TORCH else object):
    def __init__(self, input_dim=16, hidden=64):
        if not _HAS_TORCH:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        return self.net(x).squeeze(-1)


class ReRanker:
    def __init__(self, model_path: str = None, input_dim: int = 16, device: str = None):
        self.input_dim = input_dim
        self.device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._last_mtime = None
        self.feature_keys = None  # sẽ lấy từ checkpoint; nếu không có dùng fallback
        self.bad_keys = {"re_rank_score", "debug", "source"}  # không cho vào feature

        if _HAS_TORCH:
            self.model = SimpleRanker(input_dim=input_dim).to(self.device)

            # load trực tiếp nếu có path
            if model_path and os.path.exists(model_path):
                try:
                    state = torch.load(model_path, map_location=self.device)
                    # lấy feature_keys nếu có trong checkpoint
                    self.feature_keys = state.get("feature_keys") if isinstance(state, dict) else None
                    # nếu state chỉ là state_dict thuần
                    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
                    self.model.load_state_dict(sd)
                    try:
                        self._last_mtime = os.path.getmtime(model_path)
                    except Exception:
                        self._last_mtime = None
                except Exception:
                    pass
            else:
                # auto-discovery meta json
                try:
                    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    models_dir = os.path.join(base_dir, 'models')
                    if os.path.isdir(models_dir):
                        metas = [os.path.join(models_dir, p)
                                 for p in os.listdir(models_dir)
                                 if p.startswith('re_ranker_') and p.endswith('.json')]
                        metas = sorted(metas, key=lambda x: os.path.getmtime(x), reverse=True)
                        if metas:
                            with open(metas[0], 'r', encoding='utf-8') as f:
                                meta = json.load(f)
                                mp = meta.get('model_path')
                                if mp and os.path.exists(mp):
                                    state = torch.load(mp, map_location=self.device)
                                    self.feature_keys = state.get("feature_keys") if isinstance(state, dict) else None
                                    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
                                    self.model.load_state_dict(sd)
                                    self.model_path = os.path.abspath(mp)
                                    try:
                                        self._last_mtime = os.path.getmtime(mp)
                                    except Exception:
                                        self._last_mtime = None
                                    logging.getLogger(__name__).info("Auto-loaded re-ranker: %s", mp)
                except Exception:
                    pass

            # fallback feature_keys (phải KHỚP lúc train)
            if not self.feature_keys:
                self.feature_keys = [
                    "similarity", "text_sim", "image_sim", "rank",
                    "brand_match", "has_ocr", "is_primary",
                    "keywords_len", "colors_len", "score"
                ]
            self.model.eval()
        else:
            self.model = None

    def _feat_to_vec(self, f: Dict) -> List[float]:
        fd = dict(f or {})
        # loại bỏ khóa rác & convert bool -> float
        for k in list(fd.keys()):
            if k in self.bad_keys:
                fd.pop(k, None)
        for k, v in list(fd.items()):
            if isinstance(v, bool):
                fd[k] = 1.0 if v else 0.0
            # count mảng phổ biến nếu user đẩy raw list vào
            if k in ("keywords", "colors") and isinstance(v, (list, tuple)):
                fd[k + "_len"] = float(len(v))

        # ép đúng thứ tự key
        x = [float(fd.get(k, 0.0)) for k in self.feature_keys]
        # pad/trim về input_dim
        if len(x) < self.input_dim:
            x += [0.0] * (self.input_dim - len(x))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]
        return x

    def score_candidates(self, feature_list: List[Dict]) -> List[float]:
        if not feature_list:
            return []
        try:
            vecs = [self._feat_to_vec(f) for f in feature_list]
            try:
                self.reload_if_needed()
            except Exception:
                pass

            if _HAS_TORCH and self.model is not None:
                t = torch.from_numpy(np.array(vecs, dtype=np.float32)).to(self.device)
                with torch.no_grad():
                    s = self.model(t).cpu().numpy().tolist()
                return [float(x) for x in s]
            else:
                # fallback: similarity/score
                out = []
                for f in feature_list:
                    try:
                        out.append(float((f or {}).get("similarity") or (f or {}).get("score") or 0.0))
                    except Exception:
                        out.append(0.0)
                return out
        except Exception:
            out = []
            for f in feature_list:
                try:
                    out.append(float((f or {}).get("similarity") or (f or {}).get("score") or 0.0))
                except Exception:
                    out.append(0.0)
            return out

    def save(self, path: str):
        if not _HAS_TORCH or self.model is None:
            raise RuntimeError("torch not available or model not initialized")
        torch.save({"state_dict": self.model.state_dict(), "feature_keys": self.feature_keys}, path)
        self.model_path = path
        try:
            self._last_mtime = os.path.getmtime(path)
        except Exception:
            self._last_mtime = None
        logging.getLogger(__name__).info("Saved re-ranker model to %s", path)

    def load(self, path: str):
        if not _HAS_TORCH or self.model is None:
            raise RuntimeError("torch not available or model not initialized")
        state = torch.load(path, map_location=self.device)
        self.feature_keys = state.get("feature_keys") if isinstance(state, dict) else self.feature_keys
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.model.load_state_dict(sd)
        self.model.eval()
        self.model_path = path
        try:
            self._last_mtime = os.path.getmtime(path)
        except Exception:
            self._last_mtime = None
        logging.getLogger(__name__).info("Loaded re-ranker from %s", path)

    def reload_if_needed(self, force: bool = False) -> bool:
        if not _HAS_TORCH or self.model is None or not self.model_path:
            return False
        if force:
            self.load(self.model_path)
            return True
        if os.path.exists(self.model_path):
            m = os.path.getmtime(self.model_path)
            if self._last_mtime is None or m > self._last_mtime:
                self.load(self.model_path)
                return True
        return False
