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
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import csv

# ===== Model định nghĩa lại =====
class RetrievalNet(nn.Module):
    def __init__(self, backbone="resnet101", emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
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

# ===== Fine-tuning =====
def fine_tune_model(model, train_loader, device, epochs=5, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_loader)}")
    model.eval()
    return model

# ===== Load model =====
def load_model(checkpoint_path="out-resnet101-model/finetuned_resnet101.pt", device="cuda", fine_tune=False, train_loader=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(emb_dim=512)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Warning: Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")
            print("  ✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            print(f"  Using pretrained weights instead")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using pretrained weights")
    model.to(device)
    if fine_tune and train_loader is not None:
        print("Fine-tuning model...")
        model = fine_tune_model(model, train_loader, device)
        torch.save({'model': model.state_dict()}, checkpoint_path)
    model.eval()
    return model

# ===== Giảm chiều vector =====
def reduce_embedding(embedding, n_components=256):
    pca = PCA(n_components=n_components)
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    return pca.fit_transform(embedding).astype("float32")

# ===== Preprocess ảnh =====
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
            if w > 50 and h > 50:
                cropped = img[y:y+h, x:x+w]
            else:
                cropped = img
        else:
            cropped = img
        cropped = cv2.resize(cropped, (256, 256))
        return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Preprocessing error for {image_path}: {e}")
        return Image.open(image_path).convert("RGB")

# ===== Dataset class (MOVED OUTSIDE) =====
class ImageEmbeddingDataset(Dataset):
    """Dataset for image embedding extraction"""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = preprocess_image(img_path)
            if img is not None:
                return self.transform(img), img_path
            return None, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, img_path

# ===== Custom collate function to handle None values =====
def collate_fn_filter_none(batch):
    """Filter out None values from batch"""
    batch = [(img, path) for img, path in batch if img is not None]
    if len(batch) == 0:
        return [], []
    images, paths = zip(*batch)
    images = torch.stack(images)
    return images, list(paths)

# ===== Trích xuất embedding =====
def extract_embedding(model, image_paths, device=None, reduce_dim=False, n_components=256, batch_size=128):
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    if device is None:
        device = next(model.parameters()).device
    
    # Define transform
    IMG_SIZE = 224
    infer_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda x: F.equalize(x) if x.mode == 'RGB' else x),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create dataset and dataloader with custom collate_fn
    dataset = ImageEmbeddingDataset(image_paths, transform=infer_tf)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn_filter_none
    )

    embeddings = []
    image_paths_list = []
    model.eval()
    with torch.no_grad():
        for images, paths in dataloader:
            if len(images) == 0:
                continue
            images = images.to(device)
            batch_embeddings = model(images).cpu().numpy().astype("float32")
            faiss.normalize_L2(batch_embeddings)
            if reduce_dim:
                batch_embeddings = reduce_embedding(batch_embeddings, n_components)
            else:
                batch_embeddings = batch_embeddings.tolist()
            embeddings.extend(batch_embeddings)
            image_paths_list.extend(paths)

    return embeddings if len(embeddings) > 1 else embeddings[0] if embeddings else None

# ===== Xây dựng FAISS Index =====
def build_faiss_index(vectors, dim=512, m=64, nlist=None, nprobe=10):
    if len(vectors) == 0:
        raise ValueError("No vectors provided to build FAISS index")
    nlist = nlist or max(50, len(vectors) // 10)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    vectors_arr = np.array(vectors).astype("float32")
    faiss.normalize_L2(vectors_arr)
    index.train(vectors_arr)
    index.add(vectors_arr)
    index.nprobe = nprobe
    faiss.write_index(index, "faiss_index.index")
    return index

# ===== Tìm kiếm với FAISS =====
def search_with_faiss(index, query_vector, k=20, rerank=False, embeddings_for_rerank=None):
    """
    Search FAISS index with optional re-ranking.
    
    Args:
        index: FAISS index
        query_vector: Query embedding vector
        k: Number of results to return
        rerank: Whether to rerank using cosine similarity
        embeddings_for_rerank: Array of embeddings for reranking (required if rerank=True)
    
    Returns:
        distances, indices: Arrays of distances and indices
    """
    if index.ntotal == 0:
        raise ValueError("FAISS index is empty")
    query = np.array([query_vector]).astype("float32")
    faiss.normalize_L2(query)
    
    search_k = k * 2 if rerank else k
    distances, indices = index.search(query, search_k)
    
    if rerank and embeddings_for_rerank is not None:
        valid_mask = (indices[0] != -1) & (indices[0] < len(embeddings_for_rerank))
        valid_indices = indices[0][valid_mask]
        
        if len(valid_indices) > 0:
            embeddings = embeddings_for_rerank[valid_indices]
            similarities = cosine_similarity([query_vector], embeddings)[0]
            sorted_idx = np.argsort(-similarities)[:k]
            distances = distances[0][valid_mask][sorted_idx]
            indices = valid_indices[sorted_idx]
        else:
            distances = distances[0][:k]
            indices = indices[0][:k]
    else:
        distances = distances[0]
        indices = indices[0]
    
    return distances, indices

# ===== Dataset cho All Images =====
class AllImagesDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = preprocess_image(img_path)
            if image is None:
                return None, img_path
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, img_path

# ===== Main execution =====
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = "out-resnet101-model/finetuned_resnet101.pt"
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    
    csv_path = "retrieval_triplet_dataset.csv"
    all_image_paths = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_image_paths.add(row['anchor_image_path'])
                all_image_paths.add(row['positive_image_path'])
                all_image_paths.add(row['negative_image_path'])
        all_image_paths = list(all_image_paths)
        print(f"Loaded {len(all_image_paths)} unique image paths from {csv_path}")
    else:
        print(f"Warning: {csv_path} not found, using empty list")
        all_image_paths = []
    
    if not all_image_paths:
        print("No images to process. Exiting.")
        exit(0)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    all_images_dataset = AllImagesDataset(all_image_paths, transform=transform)
    all_images_dataloader = DataLoader(
        all_images_dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn_filter_none
    )

    print("Generating embeddings for all images...")
    all_embeddings = []
    image_paths_list = []

    model.eval()
    with torch.no_grad():
        for images, paths in all_images_dataloader:
            if len(images) == 0:
                continue
            images = images.to(device)
            batch_embeddings = model(images).cpu().numpy()
            all_embeddings.append(batch_embeddings)
            image_paths_list.extend(paths)

    if not all_embeddings:
        print("No embeddings generated. Exiting.")
        exit(0)

    all_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings generated: {all_embeddings.shape}")

    dim = 512
    quantizer = faiss.IndexFlatL2(dim)
    nlist = max(50, len(all_embeddings) // 10)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 16, 8)
    faiss.normalize_L2(all_embeddings)
    index.train(all_embeddings.astype('float32'))
    index.add(all_embeddings.astype('float32'))
    index.nprobe = 10
    faiss.write_index(index, "faiss_index.index")
    print(f"Faiss index created with {index.ntotal} embeddings.")

    k = 5
    query_emb = extract_embedding(model, all_image_paths[0], device=device)
    if query_emb is not None:
        distances, indices = search_with_faiss(
            index, 
            query_emb, 
            k=k, 
            rerank=True, 
            embeddings_for_rerank=all_embeddings
        )
        print(f"Found {k} nearest neighbors after re-ranking.")
        print("Most similar images:")
        for i in range(k):
            img_index = indices[i]
            distance = distances[i]
            image_path = image_paths_list[img_index]
            print(f"Rank {i+1}: Image Path: {image_path}, Distance: {distance:.4f}")
            try:
                img = Image.open(image_path)
                print(f"  Image size: {img.size}")
            except FileNotFoundError:
                print(f"  Could not open image: {image_path}")
    else:
        print("No embedding extracted for query image.")