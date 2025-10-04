# services/resnet101.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import faiss
import numpy as np
import os
import timm
from torchvision.transforms import functional as F
from sklearn.decomposition import PCA

# ===== Model định nghĩa lại =====
class RetrievalNet(nn.Module):
    def __init__(self, backbone="resnet101", emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
        in_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, emb_dim)
        )
    def forward(self, x, l2norm=True):
        feat = self.backbone(x)
        z = self.proj(feat)
        if l2norm:
            z = nn.functional.normalize(z, p=2, dim=1)
        return z

# ===== Fine-tuning (Placeholder) =====
def fine_tune_model(model, train_loader, device, epochs=5, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0)
    model.train()
    for epoch in range(epochs):
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    model.eval()
    return model

# ===== Load model =====
def load_model(checkpoint_path="out-resnet101-triplet/best.pt", device="cuda", fine_tune=False, train_loader=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = RetrievalNet(emb_dim=512)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    if fine_tune and train_loader:
        model = fine_tune_model(model, train_loader, device)
    model.eval()
    return model

# ===== Giảm chiều vector =====
def reduce_embedding(embedding, n_components=256):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embedding.reshape(1, -1))[0].tolist()

# ===== Trích xuất embedding =====
def extract_embedding(model, image_path, device=None, reduce_dim=False, n_components=256):
    if device is None:
        device = next(model.parameters()).device
    try:
        IMG_SIZE = 224
        infer_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: F.equalize(x) if x.mode == 'RGB' else x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = infer_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img).cpu().numpy().astype("float32")
            faiss.normalize_L2(embedding)
            if reduce_dim:
                embedding = reduce_embedding(embedding, n_components)
            else:
                embedding = embedding[0].tolist()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ===== Xây dựng FAISS Index =====
def build_faiss_index(vectors, dim=512, m=64, nlist=100, nprobe=10):
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index.train(np.array(vectors))
    index.add(np.array(vectors))
    index.nprobe = nprobe
    faiss.write_index(index, "faiss_index.index")
    return index

# ===== Tìm kiếm với FAISS =====
def search_with_faiss(index, query_vector, k=20):
    query = np.array([query_vector])
    distances, indices = index.search(query, k)
    return distances[0], indices[0]

# ...existing code...

if __name__ == "__main__":
    import argparse
    import os
    import torch
    import traceback

    parser = argparse.ArgumentParser(description="Check resnet101 checkpoint load")
    parser.add_argument("--ckpt", "-c", default="out-resnet101-triplet/best.pt", help="Path to checkpoint")
    parser.add_argument("--device", "-d", default="cpu", help="Device to load model on (cpu/cuda)")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    print("Checking checkpoint:", ckpt_path)
    if not os.path.exists(ckpt_path):
        print("❌ Checkpoint file not found:", ckpt_path)
        raise SystemExit(1)

    try:
        # load checkpoint without mapping to a model first
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # determine state_dict inside checkpoint
        sd = ckpt.get("model") if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        print("Checkpoint loaded. type:", type(ckpt), "contains model state_dict:", isinstance(sd, dict))
        # instantiate model and try load_state_dict (non-strict to see missing/unexpected keys)
        model = RetrievalNet(emb_dim=512)
        res = model.load_state_dict(sd, strict=False)
        # PyTorch returns a named tuple with missing_keys/unexpected_keys
        missing = getattr(res, "missing_keys", None)
        unexpected = getattr(res, "unexpected_keys", None)
        print("load_state_dict result -> missing keys:", len(missing) if missing is not None else None,
              ", unexpected keys:", len(unexpected) if unexpected is not None else None)
        if missing:
            print("Some missing keys (first 10):", missing[:10])
        if unexpected:
            print("Some unexpected keys (first 10):", unexpected[:10])

        # quick forward pass
        dev = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        model.to(dev)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            out = model(dummy)
        print("Forward OK. Output shape:", getattr(out, "shape", None))
        print("✅ Checkpoint OK and model forward-pass succeeded.")
    except Exception as e:
        print("❌ Failed to load/check checkpoint:", type(e).__name__, e)
        traceback.print_exc()
        raise