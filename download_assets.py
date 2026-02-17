import os
import sys
import zipfile
import requests
from tqdm import tqdm

def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, "wb") as f, tqdm(
        total=total_size, unit='iB', unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    print(f"Finished downloading {destination}")

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Finished unzipping {extract_to}")

def setup_assets():
    # Define paths
    backend_dir = "backend"
    models_dir = os.path.join(backend_dir, "models")
    omni_dir = os.path.join(backend_dir, "OmniParser", "weights")
    
    assets = [
        {
            "name": "Vosk Model",
            "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            "dest": os.path.join(models_dir, "vosk-model-en-us-0.22.zip"),
            "extract_to": models_dir,
            "type": "zip"
        },
        {
            "name": "Piper Voice (Ryan High)",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
            "dest": os.path.join(models_dir, "piper", "en_US-ryan-high.onnx"),
            "type": "file"
        },
        {
            "name": "Piper Config",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
            "dest": os.path.join(models_dir, "piper", "en_US-ryan-high.onnx.json"),
            "type": "file"
        },
        {
            "name": "OmniParser Model",
            "url": "https://huggingface.co/microsoft/OmniParser/resolve/main/icon_caption_florence/model.safetensors",
            "dest": os.path.join(omni_dir, "icon_caption_florence", "model.safetensors"),
            "type": "file"
        }
    ]

    for asset in assets:
        if not os.path.exists(asset["dest"]):
            download_file(asset["url"], asset["dest"])
            if asset.get("type") == "zip":
                unzip_file(asset["dest"], asset["extract_to"])
                os.remove(asset["dest"])
        else:
            print(f"Asset '{asset['name']}' already exists at {asset['dest']}. Skipping.")

if __name__ == "__main__":
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Installing required packages: requests, tqdm...")
        os.system(f"{sys.executable} -m pip install requests tqdm")
        import requests
        from tqdm import tqdm
    
    setup_assets()
    print("\nAll assets are set up correctly!")
