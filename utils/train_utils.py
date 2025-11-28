from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    patience: int = 5
    model_name: str = "resnet18"
    img_size: int = 224
    batch_size: int = 64
    seed: int = 42


@dataclass
class TrainState:
    epoch: int
    best_val_acc: float
    epochs_no_improve: int


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def evaluate(model: nn.Module, dl: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += yb.size(0)
            ys.append(yb.cpu().numpy())
            ps.append(logits.argmax(dim=1).cpu().numpy())
    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    return avg_loss, acc, y_true, y_pred


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_latency(model: nn.Module, device: torch.device, img_size: int = 224, warmup: int = 10, iters: int = 50, batch_size: int = 1) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start
    avg_ms = (elapsed / iters) * 1000.0 / batch_size
    return avg_ms


def train_model(
    model: nn.Module,
    loaders,
    device: torch.device,
    cfg: TrainConfig,
    class_names: list[str],
    out_dir: Path,
):
    out_dir = Path(out_dir)
    models_dir = out_dir / 'models'
    metrics_dir = out_dir / 'metrics'
    plots_dir = out_dir / 'plots'
    tb_dir = out_dir / 'tb'
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=loaders.class_weights.to(device) if loaders.class_weights is not None else None)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == 'cuda'))

    state = TrainState(epoch=0, best_val_acc=0.0, epochs_no_improve=0)

    # Parameter counts and baseline latency
    total_params, trainable_params = count_parameters(model)
    latency_ms = benchmark_latency(model, device, img_size=cfg.img_size, batch_size=1)

    # CSV log
    log_csv = metrics_dir / 'train_log.csv'
    with open(log_csv, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,lr\n')

    best_path = models_dir / f'{cfg.model_name}_best.pt'

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for xb, yb in loaders.train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == 'cuda')):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * yb.size(0)
            running_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)

        val_loss, val_acc, y_true_val, y_pred_val = (0.0, 0.0, np.array([]), np.array([]))
        if loaders.val is not None:
            val_loss, val_acc, y_true_val, y_pred_val = evaluate(model, loaders.val, criterion, device)

        # Step LR after epoch
        scheduler.step()

        # Write log row
        with open(log_csv, 'a') as f:
            lr_curr = scheduler.get_last_lr()[0]
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.4f},{val_loss:.6f},{val_acc:.4f},{lr_curr:.6f}\n")

        # Early stopping on val_acc
        improved = val_acc > state.best_val_acc
        if improved:
            state.best_val_acc = val_acc
            state.epochs_no_improve = 0
            torch.save({'model_state': model.state_dict(), 'cfg': asdict(cfg)}, best_path)
        else:
            state.epochs_no_improve += 1

        print(f"Epoch {epoch}/{cfg.epochs} - train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if cfg.patience and state.epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
            break

    # Load best for evaluation
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])

    # Final evaluation on val and test
    final = {}
    if loaders.val is not None:
        v_loss, v_acc, y_true_v, y_pred_v = evaluate(model, loaders.val, criterion, device)
        final['val'] = {'loss': float(v_loss), 'acc': float(v_acc)}
        if y_true_v.size > 0:
            report_v = classification_report(y_true_v, y_pred_v, output_dict=True)
            save_json(report_v, metrics_dir / 'classification_report_val.json')
            plot_confusion_matrix(y_true_v, y_pred_v, class_names=class_names, out_path=plots_dir / 'confusion_matrix_val.png')

    if loaders.test is not None:
        t_loss, t_acc, y_true_t, y_pred_t = evaluate(model, loaders.test, criterion, device)
        final['test'] = {'loss': float(t_loss), 'acc': float(t_acc)}
        if y_true_t.size > 0:
            report_t = classification_report(y_true_t, y_pred_t, output_dict=True)
            save_json(report_t, metrics_dir / 'classification_report_test.json')
            plot_confusion_matrix(y_true_t, y_pred_t, class_names=class_names, out_path=plots_dir / 'confusion_matrix_test.png')

    summary = {
        'model': cfg.model_name,
        'img_size': cfg.img_size,
        'epochs_trained': epoch,
        'best_val_acc': state.best_val_acc,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'latency_ms_per_image_bs1': latency_ms,
        'final': final,
    }
    save_json(summary, metrics_dir / 'last_train_summary.json')
    return summary
