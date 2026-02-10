import wandb
import os

# Your API Key and login
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

def download_steering_vectors_flat(entity, project, artifact_base_name, versions):
    api = wandb.Api()
    downloaded_files = []

    for v in versions:
        artifact_path = f"{entity}/{project}/{artifact_base_name}:v{v}"
        print(f"Downloading {artifact_path}...")
        
        try:
            artifact = api.artifact(artifact_path)
            datadir = artifact.download(root=".") 
            downloaded_files.append(datadir)
        except Exception as e:
            print(f"Error downloading v{v}: {e}")

    return downloaded_files

ENTITY = "yosefnuragaw" 
PROJECT = "BiPO-Gemma-3-Power-Seeking"
ARTIFACT_NAME = "power-seeking-Layers_13-j5mndz4z_steering-vec-layer13"

versions_to_get = list(range(20)) 

paths = download_steering_vectors_flat(ENTITY, PROJECT, ARTIFACT_NAME, versions_to_get)

print(f"\nFinished. Files are in: {os.getcwd()}")