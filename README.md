# Logo Recognition PoC

Recognizes company logos from the HalalDamu database using OpenCLIP + FAISS.

## Setup

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr  # for OCR

# 1. Download logos
python fetch_companies.py

# 2. Build FAISS index
python build_index.py

# 3. Run API
python app.py  # serves on :8090

# 4. Test
python test_match.py data/logos/1/logo.png
```

## API

**POST /match** — multipart file upload, returns JSON with candidates and scores.
