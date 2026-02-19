#!/usr/bin/env python3
"""Test the /match endpoint with a sample logo."""
import requests, sys, json

url = "http://localhost:8090/match"
img_path = sys.argv[1] if len(sys.argv) > 1 else "data/logos/1/logo.png"

print(f"Testing with: {img_path}")
with open(img_path, "rb") as f:
    resp = requests.post(url, files={"file": f})

print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
