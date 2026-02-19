#!/usr/bin/env python3
"""Build FAISS index from logo embeddings using OpenCLIP."""
import json, numpy as np, faiss, time
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import open_clip, torch

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index_artifacts"
INDEX_DIR.mkdir(exist_ok=True)

def load_and_preprocess(path, preprocess):
    """Load image and return preprocessed tensor."""
    img = Image.open(path).convert("RGB")
    return img, preprocess(img).unsqueeze(0)

def augment_image(img, preprocess):
    """Generate augmented versions of an image."""
    augmented = []
    # Rotation variants
    for angle in [-5, 5]:
        aug = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        augmented.append(preprocess(aug).unsqueeze(0))
    # Brightness variants
    for factor in [0.8, 1.2]:
        aug = ImageEnhance.Brightness(img).enhance(factor)
        augmented.append(preprocess(aug).unsqueeze(0))
    # Contrast variant
    aug = ImageEnhance.Contrast(img).enhance(1.3)
    augmented.append(preprocess(aug).unsqueeze(0))
    return augmented

def main():
    print("Loading OpenCLIP ViT-B/32...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    
    companies = json.loads((DATA_DIR / "companies.json").read_text())
    print(f"Processing {len(companies)} companies...")
    
    all_embeddings = []
    metadata = []
    
    t0 = time.time()
    for i, c in enumerate(companies):
        for lp in c["logo_paths"]:
            logo_path = DATA_DIR / lp
            if not logo_path.exists():
                continue
            try:
                img, tensor = load_and_preprocess(logo_path, preprocess)
            except Exception as e:
                print(f"  Skip {c['company_id']}: {e}")
                continue
            
            # Original + augmented
            tensors = [tensor] + augment_image(img, preprocess)
            
            with torch.no_grad():
                batch = torch.cat(tensors, dim=0)
                embs = model.encode_image(batch)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                embs = embs.numpy().astype(np.float32)
            
            for j, emb in enumerate(embs):
                all_embeddings.append(emb)
                metadata.append({
                    "company_id": c["company_id"],
                    "company_name": c["company_name"],
                    "aug": "original" if j == 0 else f"aug_{j}"
                })
        
        if (i + 1) % 100 == 0:
            print(f"  Embedded {i+1}/{len(companies)} ({time.time()-t0:.1f}s)")
    
    if not all_embeddings:
        print("No embeddings generated!")
        return
    
    embeddings = np.stack(all_embeddings).astype(np.float32)
    print(f"Building FAISS index: {embeddings.shape}")
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)
    with open(INDEX_DIR / "metadata.jsonl", "w") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    
    print(f"Done in {time.time()-t0:.1f}s: {len(metadata)} vectors indexed")

if __name__ == "__main__":
    main()
