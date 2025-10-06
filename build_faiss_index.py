"""
Build FAISS index from all sku_images in database
Based on Copy_of_finetune_ResNet_101.ipynb workflow
"""
import torch
import numpy as np
import faiss
import os
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.db_utils import q, exec_sql
from services.resnet101 import RetrievalNet, load_model, collate_fn_filter_none

# Constants
EMB_DIM = 512
BATCH_SIZE = 128
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

class DatabaseImagesDataset(Dataset):
    """Dataset for loading images from database"""
    def __init__(self, image_records, upload_dir, transform=None):
        """
        Args:
            image_records: List of (id, sku_id, image_path) tuples
            upload_dir: Root directory for images
            transform: Image transforms
        """
        self.records = image_records
        self.upload_dir = upload_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, sku_id, image_path = self.records[idx]
        full_path = os.path.join(self.upload_dir, image_path)
        
        try:
            image = Image.open(full_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, (img_id, sku_id, image_path)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, (img_id, sku_id, image_path)

def build_faiss_index_from_db(checkpoint_path="out-resnet101-model/finetuned_resnet101.pt"):
    """
    Build FAISS index from all sku_images in database
    Follows the workflow from notebook Step 16-17
    """
    print("="*80)
    print("Building FAISS Index from Database Images")
    print("="*80)
    
    # 1. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    
    # 2. Get all images from database
    rows = q("""
        SELECT id, sku_id, image_path 
        FROM sku_images 
        WHERE image_path IS NOT NULL
        ORDER BY sku_id, is_primary DESC, id
    """)
    
    if not rows:
        print("❌ No images found in database")
        return None
    
    print(f"Found {len(rows)} images in database")
    
    # 3. Create dataset and dataloader (same as notebook Step 16)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = DatabaseImagesDataset(rows, UPLOAD_DIR, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,
        collate_fn=collate_fn_filter_none
    )
    
    # 4. Generate embeddings for all images (same as notebook)
    print("Generating embeddings for all images...")
    all_embeddings = []
    metadata_list = []  # Store (img_id, sku_id, image_path)
    
    with torch.no_grad():
        for batch_idx, (images, metadata) in enumerate(dataloader):
            if len(images) == 0:
                continue
            
            images = images.to(device)
            batch_embeddings = model(images).cpu().numpy()
            all_embeddings.append(batch_embeddings)
            metadata_list.extend(metadata)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {len(metadata_list)} images...")
    
    if not all_embeddings:
        print("❌ No embeddings generated")
        return None
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"✅ Embeddings generated: {all_embeddings.shape}")
    
    # 5. Build FAISS index (same as notebook Step 17)
    all_embeddings = all_embeddings.astype('float32')
    
    # Use IndexFlatL2 (L2 distance) - same as notebook
    index = faiss.IndexFlatL2(EMB_DIM)
    index.add(all_embeddings)
    
    print(f"✅ FAISS index created with {index.ntotal} embeddings")
    
    # 6. Save index and metadata
    index_path = "faiss_index.index"
    metadata_path = "faiss_metadata.npy"
    
    faiss.write_index(index, index_path)
    np.save(metadata_path, metadata_list)
    
    print(f"✅ Saved FAISS index to {index_path}")
    print(f"✅ Saved metadata to {metadata_path}")
    
    return index, metadata_list

def search_image_with_faiss(query_image_path, k=5, index_path="faiss_index.index", metadata_path="faiss_metadata.npy"):
    """
    Search for similar images using FAISS index
    Follows notebook Step 18-19
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device=device)
    model.eval()
    
    # Load FAISS index and metadata
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("❌ FAISS index not found. Run build_faiss_index_from_db() first.")
        return []
    
    index = faiss.read_index(index_path)
    metadata_list = np.load(metadata_path, allow_pickle=True)
    
    print(f"Loaded FAISS index with {index.ntotal} embeddings")
    
    # Load and transform query image (same as notebook Step 13)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        query_image = Image.open(query_image_path).convert('RGB')
        transformed_image = transform(query_image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading query image: {e}")
        return []
    
    # Generate embedding for query image (same as notebook Step 15)
    with torch.no_grad():
        query_embedding = model(transformed_image).cpu().numpy().astype('float32')
    
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Search FAISS index (same as notebook Step 18)
    distances, indices = index.search(query_embedding, k)
    
    print(f"Found {k} nearest neighbors:")
    
    results = []
    for i in range(k):
        idx = indices[0][i]
        distance = distances[0][i]
        
        if 0 <= idx < len(metadata_list):
            img_id, sku_id, image_path = metadata_list[idx]
            print(f"Rank {i+1}: SKU {sku_id} | {image_path} | Distance: {distance:.4f}")
            
            results.append({
                "rank": i+1,
                "img_id": img_id,
                "sku_id": sku_id,
                "image_path": image_path,
                "distance": float(distance),
                "score": 1.0 / (1.0 + distance)  # Convert distance to similarity score
            })
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index or search images")
    parser.add_argument("--build", action="store_true", help="Build FAISS index from database")
    parser.add_argument("--search", type=str, help="Path to query image for search")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--checkpoint", type=str, default="out-resnet101-model/finetuned_resnet101.pt", 
                       help="Path to model checkpoint")
    args = parser.parse_args()
    
    if args.build:
        print("Building FAISS index...")
        build_faiss_index_from_db(checkpoint_path=args.checkpoint)
    elif args.search:
        print(f"Searching for similar images to: {args.search}")
        results = search_image_with_faiss(args.search, k=args.k)
        if results:
            print(f"\n✅ Found {len(results)} similar images")
    else:
        parser.print_help()