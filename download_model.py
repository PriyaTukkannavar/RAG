import os
import requests

# Model repo and branch
base_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

# Files to download with correct folder structure
files_to_download = [
    # Root
    "config.json",
    "modules.json",

    # Transformer module
    "0_Transformer/config.json",
    "0_Transformer/pytorch_model.bin",
    "0_Transformer/tokenizer.json",
    "0_Transformer/tokenizer_config.json",
    "0_Transformer/vocab.txt",

    # Pooling module
    "1_Pooling/config.json"
]

# Local folder to save model
model_dir = r"C:\Users\v-priya.tukkannavar\Downloads\RAG1\all-MiniLM-L6-v2"

# Ensure directories exist
for file_path in files_to_download:
    local_path = os.path.join(model_dir, file_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Download file without SSL verification
    url = f"{base_url}/{file_path}"
    print(f"Downloading: {url}")
    resp = requests.get(url, verify=False, stream=True)  # bypass SSL verification
    if resp.status_code == 200:
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Saved: {local_path}")
    else:
        print(f"❌ Failed to download {url} - Status {resp.status_code}")

print("\n✅ Model download complete!")
