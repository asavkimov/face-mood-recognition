#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

# Ensure project root is on sys.path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from utils.datasets import make_dataloaders, EMOTIONS
from utils.train_utils import TrainConfig, set_seed, train_model
from models import ResNet18Baseline, SmallCNN


def build_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name in {"resnet", "resnet18", "rn18"}:
        m = ResNet18Baseline(num_classes=num_classes, pretrained=True, dropout=0.2)
        # For transfer learning: freeze backbone initially, then unfreeze in second phase
        return m
    elif name in {"smallcnn", "cnn"}:
        return SmallCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def write_readme_note(summary_path: Path):
    # Append a short baseline run note to README.md
    try:
        repo_root = Path(__file__).resolve().parents[1]
        readme = repo_root / 'README.md'
        if not readme.exists():
            return
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        note = []
        note.append("\n---\n\n## Baseline Training (Auto)\n")
        note.append(f"- Model: `{summary.get('model')}`; Image size: `{summary.get('img_size')}`\n")
        best_val = summary.get('best_val_acc')
        if isinstance(best_val, float):
            note.append(f"- Best Val Accuracy: `{best_val:.4f}`\n")
        final = summary.get('final', {})
        if 'test' in final:
            note.append(f"- Test Accuracy: `{final['test'].get('acc', 0):.4f}`\n")
        note.append("- Outputs: see `outputs/models/`, `outputs/metrics/`, `outputs/plots/`.\n")
        with open(readme, 'a') as f:
            f.write("".join(note))
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="Train a baseline image classifier on FER2013")
    p.add_argument('--data', type=str, default='data/processed/fer2013', help='Processed FER2013 directory')
    p.add_argument('--out', type=str, default='outputs', help='Outputs root directory')
    p.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'smallcnn'])
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--freeze_epochs', type=int, default=2, help='Warm-up epochs with frozen backbone (ResNet only)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit_per_class', type=int, default=0, help='Limit samples per class in train split (for quick smoke tests)')
    p.add_argument('--no_amp', action='store_true', help='Disable mixed precision')

    args = p.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = make_dataloaders(proc_dir=args.data, img_size=args.img_size, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=loaders.num_classes)
    model = model.to(device)

    out_root = Path(args.out)

    # Phase 1: optional freeze backbone
    remaining_epochs = args.epochs
    if args.model == 'resnet18' and args.freeze_epochs > 0:
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
        cfg1 = TrainConfig(
            epochs=min(args.freeze_epochs, args.epochs),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=not args.no_amp,
            patience=max(2, min(args.freeze_epochs, 3)),
            model_name=f'{args.model}_frozen',
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summary1 = train_model(model, loaders, device, cfg1, EMOTIONS, out_root)
        remaining_epochs = max(0, args.epochs - cfg1.epochs)
        # Unfreeze for fine-tuning
        if hasattr(model, 'unfreeze_all'):
            model.unfreeze_all()

    # Phase 2: fine-tune
    if remaining_epochs > 0:
        cfg2 = TrainConfig(
            epochs=remaining_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            amp=not args.no_amp,
            patience=5,
            model_name=args.model,
            img_size=args.img_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summary2 = train_model(model, loaders, device, cfg2, EMOTIONS, out_root)
        # Append to README (non-fatal on failure)
        metrics_dir = out_root / 'metrics'
        write_readme_note(metrics_dir / 'last_train_summary.json')

    print("Training complete. See outputs under:", out_root)


if __name__ == '__main__':
    main()
