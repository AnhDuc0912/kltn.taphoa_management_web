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
import cv2

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
        total_loss = 0
        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_loader)}")
    model.eval()
    return model

# ===== Load model =====
def load_model(checkpoint_path="out-resnet101-triplet/best.pt", device="cuda", fine_tune=False, train_loader=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(emb_dim=512)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random weights")
    model.to(device)
    if fine_tune and train_loader is not None:
        model = fine_tune_model(model, train_loader, device)
    model.eval()
    return model

# ===== Giảm chiều vector =====
def reduce_embedding(embedding, n_components=256):
    pca = PCA(n_components=n_components)
    if len(embedding.shape) > 1 and embedding.shape[0] > 1:
        return pca.fit_transform(embedding).astype("float32")
    return pca.fit_transform(embedding.reshape(1, -1)).astype("float32").flatten()

# ===== Preprocess ảnh để tập trung vào vật thể =====
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Đảm bảo kích thước crop không quá nhỏ
            if w > 50 and h > 50:  # Ngưỡng tối thiểu để tránh noise
                cropped = img[y:y+h, x:x+w]
            else:
                cropped = img
        else:
            cropped = img
        
        cropped = cv2.resize(cropped, (256, 256))
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Preprocessing error for {image_path}: {e}")
        return Image.open(image_path).convert("RGB")  # Fallback to original

# ===== Trích xuất embedding =====
def extract_embedding(model, image_path, device=None, reduce_dim=False, n_components=256, batch_size=32):
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
        img = preprocess_image(image_path)
        img = infer_tf(img).unsqueeze(0).to(device)
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
    if len(vectors) == 0:
        raise ValueError("No vectors provided to build FAISS index")
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    index.train(np.array(vectors).astype("float32"))
    index.add(np.array(vectors).astype("float32"))
    index.nprobe = nprobe
    faiss.write_index(index, "faiss_index.index")
    return index

# ===== Tìm kiếm với FAISS =====
def search_with_faiss(index, query_vector, k=20):
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty")
    query = np.array([query_vector]).astype("float32")
    distances, indices = index.search(query, k)
    return distances[0], indices[0]

# ===== Ví dụ sử dụng (nếu cần) =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    image_path = "E:\api_hango\flask_pgvector_shop\flask_pgvector_shop\download.jpg"
    embedding = extract_embedding(model, image_path, device=device)
    if embedding:
        # Giả lập build index (thay bằng dữ liệu thực tế)
        dummy_vectors = np.random.rand(1000, 512).astype("float32")
        index = build_faiss_index(dummy_vectors)
        distances, indices = search_with_faiss(index, embedding, k=5)
        print("Distances:", distances)
        print("Indices:", indices)