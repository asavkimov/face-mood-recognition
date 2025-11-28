from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
LABEL2IDX = {l: i for i, l in enumerate(EMOTIONS)}


def default_transforms(img_size: int = 224, train: bool = True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.2),
            T.ToTensor(),
            # Convert grayscale tensor [1,H,W] to 3 channels by repeat
            RepeatChannelsTo3(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            RepeatChannelsTo3(),
            normalize,
        ])


class RepeatChannelsTo3(torch.nn.Module):
    """If tensor has shape [1,H,W], repeat to [3,H,W]; else pass-through."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x


class FerCsvDataset(Dataset):
    def __init__(self, csv_path: Path | str, transform=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if 'path' not in self.df.columns or 'label' not in self.df.columns:
            raise ValueError(f"CSV {csv_path} must contain columns 'path' and 'label'")
        # normalize labels
        self.df['label'] = self.df['label'].str.lower().str.strip()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = row['path']
        label = row['label']
        y = LABEL2IDX[label]
        with Image.open(path) as img:
            img = img.convert('L')  # enforce single channel first (FER is grayscale)
            if self.transform is not None:
                x = self.transform(img)
            else:
                x = T.ToTensor()(img)
                x = RepeatChannelsTo3()(x)
        return x, y


@dataclass
class DataLoaders:
    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    class_weights: Optional[torch.Tensor]
    num_classes: int


def compute_class_weights(train_csv: Path | str) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df['label'].str.lower().value_counts().reindex(EMOTIONS).fillna(0).astype(int)
    counts = counts.clip(lower=1)  # avoid div by zero
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(EMOTIONS)
    return torch.tensor(weights.values, dtype=torch.float32)


def _limit_df_by_class(df: pd.DataFrame, limit_per_class: int, seed: int = 42) -> pd.DataFrame:
    if limit_per_class is None or limit_per_class <= 0:
        return df
    dfs = []
    for lbl in EMOTIONS:
        sub = df[df['label'].str.lower() == lbl]
        if len(sub) > limit_per_class:
            sub = sub.sample(n=limit_per_class, random_state=seed)
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True)


def make_dataloaders(
    proc_dir: Path | str = 'data/processed/fer2013',
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    limit_per_class: int | None = None,
    seed: int = 42,
) -> DataLoaders:
    proc_dir = Path(proc_dir)
    train_csv = proc_dir / 'train.csv'
    val_csv = proc_dir / 'val.csv'
    test_csv = proc_dir / 'test.csv'

    t_train = default_transforms(img_size, train=True)
    t_eval = default_transforms(img_size, train=False)

    # Optionally limit per class for a fast smoke test
    if limit_per_class is not None and limit_per_class > 0:
        df_train = pd.read_csv(train_csv)
        df_train = _limit_df_by_class(df_train, limit_per_class, seed)
        tmp_train_csv = proc_dir / f'train_limit_{limit_per_class}.csv'
        df_train.to_csv(tmp_train_csv, index=False)
        train_csv_to_use = tmp_train_csv
    else:
        train_csv_to_use = train_csv

    ds_train = FerCsvDataset(train_csv_to_use, transform=t_train)
    ds_val = FerCsvDataset(val_csv, transform=t_eval) if val_csv.exists() else None
    ds_test = FerCsvDataset(test_csv, transform=t_eval) if test_csv.exists() else None

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=pin_memory)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory) if ds_val is not None else None
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin_memory) if ds_test is not None else None

    class_weights = compute_class_weights(train_csv)

    return DataLoaders(train=dl_train, val=dl_val, test=dl_test,
                       class_weights=class_weights, num_classes=len(EMOTIONS))
