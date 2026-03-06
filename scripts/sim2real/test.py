from pathlib import Path
repo_root = Path(__file__).resolve().parents[3]
model_dir = repo_root / "pretrained_models" / "reach"

print(model_dir)
