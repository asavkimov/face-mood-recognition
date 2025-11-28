import os
import sys
import zipfile
from pathlib import Path


def info(msg: str):
    print(f"[download_fer2013] {msg}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / 'data' / 'raw' / 'fer2013'
    ensure_dir(raw_dir)

    # Try Python Kaggle API first
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        info("Kaggle API not available. Please install with: pip install kaggle")
        return 1

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        info("Failed to authenticate with Kaggle. Ensure ~/.kaggle/kaggle.json exists with correct permissions (600).")
        info(str(e))
        return 1

    dataset = 'msambare/fer2013'
    info(f"Downloading dataset '{dataset}' to {raw_dir} ... (this can take a few minutes)")
    try:
        api.dataset_download_files(dataset, path=str(raw_dir), unzip=False, quiet=False)
    except Exception as e:
        info("Download failed via Kaggle API.")
        info(str(e))
        return 1

    # Find the downloaded zip file (Kaggle names it <slug>.zip)
    zips = list(raw_dir.glob('*.zip'))
    if not zips:
        info("No zip file found after download. Exiting.")
        return 1

    zip_path = zips[0]
    info(f"Unzipping {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(str(raw_dir))
    except zipfile.BadZipFile:
        info("Corrupted zip. Try deleting it and rerunning.")
        return 1
    finally:
        try:
            zip_path.unlink()  # remove zip to save space
        except Exception:
            pass

    # Heuristic check
    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'
    csv_file = raw_dir / 'fer2013.csv'

    if train_dir.exists() and test_dir.exists():
        info("Images dataset detected with 'train/' and 'test/' folders.")
    elif csv_file.exists():
        info("CSV (fer2013.csv) detected. You can convert it to images or consume directly in training.")
    else:
        info("Warning: Expected 'train/' and 'test/' or 'fer2013.csv' not found after unzip. Please inspect data/raw/fer2013.")

    info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
