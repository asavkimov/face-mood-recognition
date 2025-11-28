import os
import sys
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

PLOTS_DIR = Path('outputs/plots')
RAW_DIR = Path('data/raw/fer2013')
PROC_DIR = Path('data/processed/fer2013')

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
FER_INT2LABEL = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
LABEL2IDX = {l: i for i, l in enumerate(EMOTIONS)}


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)


def gather_image_paths(root: Path):
    # Expect structure: root/train/<class>/*.png and root/test/<class>/*.png (optional val)
    splits = {}
    for split in ['train', 'val', 'validation', 'test']:
        d = root / split
        if d.exists():
            items = []
            for cls in sorted([p for p in d.iterdir() if p.is_dir()]):
                label = cls.name.lower()
                if label not in EMOTIONS:
                    continue
                for imgp in cls.glob('*'):
                    if imgp.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}:
                        items.append((str(imgp), label))
            if items:
                # normalize split name
                norm = 'val' if split in {'val', 'validation'} else split
                splits[norm] = items
    return splits


def eda_from_image_paths(splits: dict):
    # Class distribution
    all_items = []
    rows = []
    for split, items in splits.items():
        for p, lbl in items:
            rows.append({'split': split, 'label': lbl})
            all_items.append((p, lbl, split))
    if not rows:
        return None
    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, x='label', hue='split', order=EMOTIONS)
    plt.title('FER2013 class distribution by split (images)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'fer_class_distribution.png')
    plt.close()

    # Image size stats on a sample
    sample = all_items if len(all_items) <= 2000 else random.sample(all_items, 2000)
    sizes = []
    channels = []
    for p, lbl, split in sample:
        try:
            with Image.open(p) as img:
                sizes.append(img.size)  # (w, h)
                mode = img.mode  # 'L' for grayscale, 'RGB' otherwise
                channels.append({'L': 1, 'RGB': 3, 'RGBA': 4}.get(mode, 1))
        except Exception:
            continue
    if sizes:
        w = [s[0] for s in sizes]
        h = [s[1] for s in sizes]
        plt.figure(figsize=(8, 3))
        sns.histplot(w, bins=30, color='#58D68D')
        plt.title('Image widths (sample)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fer_img_width_hist.png')
        plt.close()

        plt.figure(figsize=(8, 3))
        sns.histplot(h, bins=30, color='#AF7AC5')
        plt.title('Image heights (sample)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fer_img_height_hist.png')
        plt.close()

        plt.figure(figsize=(6, 3))
        sns.countplot(x=channels)
        plt.title('Channel count (sample)')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fer_channel_counts.png')
        plt.close()

    # Sample grid per class (from train if present, else any)
    grid_items = defaultdict(list)
    for split in ['train', 'val', 'test']:
        for p, lbl, sp in all_items:
            if sp != split:
                continue
            if len(grid_items[lbl]) < 6:
                grid_items[lbl].append(p)
        if grid_items and split == 'train':
            break
    # Plot grid
    cols = 6
    rows_n = sum(1 for e in EMOTIONS if grid_items.get(e))
    if rows_n > 0:
        plt.figure(figsize=(cols * 2, rows_n * 2))
        r = 0
        for emo in EMOTIONS:
            imgs = grid_items.get(emo, [])
            if not imgs:
                continue
            for c in range(cols):
                idx = r * cols + c + 1
                plt.subplot(rows_n, cols, idx)
                try:
                    with Image.open(imgs[c % len(imgs)]) as img:
                        plt.imshow(img.convert('L'), cmap='gray')
                except Exception:
                    plt.text(0.5, 0.5, 'err', ha='center', va='center')
                plt.axis('off')
                if c == 0:
                    plt.title(emo)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fer_sample_grid.png')
        plt.close()

    return df


def write_manifest(items, out_csv: Path):
    rows = [{'path': p, 'label': lbl} for p, lbl in items]
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        os.symlink(src, dst)
    except Exception:
        # fallback to copy
        try:
            if not dst.exists():
                from shutil import copy2
                copy2(src, dst)
        except Exception:
            pass


def prepare_splits_from_images(splits: dict, val_ratio=0.1, seed=42):
    # Use provided train as base; split a val set
    train_items = splits.get('train', [])
    test_items = splits.get('test', [])

    if not train_items and not test_items:
        print('[fer] No image splits found to prepare.')
        return

    # Stratified split for val
    if train_items:
        X = np.array([p for p, _ in train_items])
        y = np.array([lbl for _, lbl in train_items])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(X, y))
        new_train = [(X[i], y[i]) for i in train_idx]
        val = [(X[i], y[i]) for i in val_idx]
    else:
        new_train, val = [], []

    # Write manifests
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(new_train, PROC_DIR / 'train.csv')
    write_manifest(val, PROC_DIR / 'val.csv')
    write_manifest(test_items, PROC_DIR / 'test.csv')

    # Also materialize split directory tree with symlinks for easy ImageFolder use
    for split_name, items in [('train', new_train), ('val', val), ('test', test_items)]:
        for p, lbl in items:
            src = Path(p)
            dst = PROC_DIR / split_name / lbl / src.name
            symlink_or_copy(src, dst)

    print('[fer] Wrote manifests and split directories under data/processed/fer2013')


# CSV-based fallback (if only fer2013.csv is available)

def eda_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    # columns: emotion, pixels, Usage
    if 'emotion' not in df.columns or 'pixels' not in df.columns:
        print('[fer] Unexpected CSV format')
        return None, None
    # map emotion ids
    df['label'] = df['emotion'].map(FER_INT2LABEL)
    if 'Usage' in df.columns:
        df['split'] = df['Usage'].map({
            'Training': 'train',
            'PublicTest': 'val',
            'PrivateTest': 'test'
        }).fillna('train')
    else:
        df['split'] = 'train'

    # Distribution plot
    plt.figure(figsize=(9, 4))
    sns.countplot(data=df, x='label', hue='split', order=EMOTIONS)
    plt.title('FER2013 class distribution by split (CSV)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'fer_class_distribution_csv.png')
    plt.close()

    # Example sample grid from CSV (by rendering a few pixels)
    sample = df.groupby('label').head(6)
    imgs = []
    for pixels in sample['pixels']:
        try:
            arr = np.fromstring(pixels, sep=' ', dtype=np.uint8)
            if arr.size != 48 * 48:
                continue
            imgs.append(arr.reshape(48, 48))
        except Exception:
            continue
    if imgs:
        cols = 6
        rows = math.ceil(len(imgs) / cols)
        plt.figure(figsize=(cols * 2, rows * 2))
        for i, img in enumerate(imgs):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'fer_sample_grid_csv.png')
        plt.close()

    return df, None


def save_images_from_csv(df: pd.DataFrame):
    # Write split directories with images decoded from CSV
    out_root = PROC_DIR
    for _, row in tqdm(df.iterrows(), total=len(df), desc='csv->images'):
        label = row['label']
        split = row['split']
        pixels = row['pixels']
        arr = np.fromstring(pixels, sep=' ', dtype=np.uint8)
        if arr.size != 48 * 48:
            continue
        img = Image.fromarray(arr.reshape(48, 48), mode='L')
        # ensure directories
        out_dir = out_root / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        # filename
        idx = row.name
        out_path = out_dir / f'{idx}.png'
        try:
            if not out_path.exists():
                img.save(out_path)
        except Exception:
            continue
    # Also create manifests from filesystem we just created
    splits = gather_image_paths(out_root)
    prepare_splits_from_images(splits, val_ratio=0.0)  # keep given splits


def main():
    ensure_dirs()
    # First, try image folders in RAW_DIR
    splits = gather_image_paths(RAW_DIR)
    if splits:
        print('[fer] Found image directories in data/raw/fer2013')
        eda_from_image_paths(splits)
        prepare_splits_from_images(splits, val_ratio=0.1)
        print('[fer] Plots saved to outputs/plots')
        return 0

    # Next, try CSV
    csv_path = RAW_DIR / 'fer2013.csv'
    if csv_path.exists():
        print('[fer] Found fer2013.csv, generating EDA and prepared images...')
        df_csv, _ = eda_from_csv(csv_path)
        if df_csv is not None:
            save_images_from_csv(df_csv)
            print('[fer] Finished preparing images and manifests from CSV')
            return 0

    print('[fer] Could not find appropriate FER2013 data. Please run scripts/download_fer2013.py first, then rerun this script.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
