"""
Phase 6 baseline promoter model v1 for promoter_atlas.

This version uses memmap-backed promoter sequences as the formal training input.

Design goals
------------
1. Keep label semantics correct.
2. Use memmap sequence assets instead of direct FASTA parsing.
3. Support masked learning for Score_S.
4. Produce reproducible logs and checkpoints.
5. Use relative paths only.

Formal targets
--------------
- Score_E
- Score_S
- Score_B

Primary keys
------------
- promoter-level: record_id
- gene-level: gene_id_base
"""

from pathlib import Path
import math
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


ROOT = Path(".")

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"

MEMMAP_PATH = ROOT / "promoters/memmap/promoter_up200_uint8.memmap"
MEMMAP_INDEX_PATH = ROOT / "promoters/memmap/promoter_up200_index.tsv"

CHECKPOINT_DIR = ROOT / "models/checkpoints"
LOG_DIR = ROOT / "models/logs"
EVAL_DIR = ROOT / "models/eval"
CARD_DIR = ROOT / "models/cards"

CHECKPOINT_PATH = CHECKPOINT_DIR / "promoter_model_v1.pt"
LOG_PATH = LOG_DIR / "train_promoter_model_v1.log"
EVAL_PATH = EVAL_DIR / "promoter_model_v1_metrics.tsv"
CARD_PATH = CARD_DIR / "promoter_model_v1_card.txt"

SEED = 20260315
BATCH_SIZE = 4096
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
AMP_ENABLED = True
SEQ_LEN = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


def r2_score_torch(pred, target):
    target_mean = target.mean()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target_mean) ** 2).sum()
    if ss_tot.item() == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def rankdata_average(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_np(x, y):
    if len(x) < 2:
        return float("nan")
    rx = rankdata_average(np.asarray(x))
    ry = rankdata_average(np.asarray(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def prepare_dataframe():
    df = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)

    required_input = [
        "record_id",
        "gene_id_base",
        "Score_E",
        "Score_S",
        "Score_B",
    ]
    required_split = ["record_id", "split"]
    required_index = ["record_id", "row_index"]

    miss_input = [c for c in required_input if c not in df.columns]
    miss_split = [c for c in required_split if c not in split.columns]
    miss_index = [c for c in required_index if c not in mm_index.columns]

    if miss_input:
        raise KeyError(f"Missing input columns: {miss_input}")
    if miss_split:
        raise KeyError(f"Missing split columns: {miss_split}")
    if miss_index:
        raise KeyError(f"Missing memmap index columns: {miss_index}")

    df = df.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")

    if df["record_id"].duplicated().any():
        dup_n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    return df


def build_targets(df):
    y = np.zeros((len(df), 3), dtype=np.float32)
    m = np.ones((len(df), 3), dtype=np.float32)

    y[:, 0] = pd.to_numeric(df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    y[:, 1] = pd.to_numeric(df["Score_S"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y[:, 2] = pd.to_numeric(df["Score_B"], errors="coerce").to_numpy(dtype=np.float32)

    s_missing = pd.to_numeric(df["Score_S"], errors="coerce").isna().to_numpy()
    m[s_missing, 1] = 0.0

    return y, m


def load_memmap_table():
    if not MEMMAP_PATH.exists():
        raise FileNotFoundError(f"Missing memmap file: {MEMMAP_PATH}")
    if not MEMMAP_INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing memmap index: {MEMMAP_INDEX_PATH}")

    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    n_rows = len(mm_index)

    mm = np.memmap(
        MEMMAP_PATH,
        dtype="uint8",
        mode="r",
        shape=(n_rows, SEQ_LEN),
    )
    return mm, mm_index


class ConvBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.backbone = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.head_e = nn.Linear(128, 1)
        self.head_s = nn.Linear(128, 1)
        self.head_b = nn.Linear(128, 1)

    def forward(self, x_idx):
        x = F.one_hot(x_idx.long(), num_classes=5)[..., :4].float()
        x = x.permute(0, 2, 1).contiguous()
        x = self.features(x).squeeze(-1)
        x = self.backbone(x)
        e = self.head_e(x)
        s = self.head_s(x)
        b = self.head_b(x)
        return torch.cat([e, s, b], dim=1)


def evaluate(model, loader):
    model.eval()

    pred_all = []
    y_all = []
    m_all = []

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            pred = model(xb)

            pred_all.append(pred.cpu())
            y_all.append(yb.cpu())
            m_all.append(mb.cpu())

    pred_all = torch.cat(pred_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    m_all = torch.cat(m_all, dim=0)

    metrics = {}
    target_names = ["Score_E", "Score_S", "Score_B"]

    for i, name in enumerate(target_names):
        mask = m_all[:, i] > 0
        yp = pred_all[mask, i]
        yt = y_all[mask, i]

        if len(yt) == 0:
            metrics[f"{name}_mse"] = float("nan")
            metrics[f"{name}_r2"] = float("nan")
            metrics[f"{name}_spearman"] = float("nan")
            continue

        mse = float(torch.mean((yp - yt) ** 2))
        r2 = r2_score_torch(yp, yt)
        sp = spearman_np(yp.numpy(), yt.numpy())

        metrics[f"{name}_mse"] = mse
        metrics[f"{name}_r2"] = r2
        metrics[f"{name}_spearman"] = sp

    return metrics


def write_log_line(handle, text):
    print(text)
    handle.write(text + "\n")
    handle.flush()


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)

    mm, mm_index = load_memmap_table()
    df = prepare_dataframe()

    max_row_index = int(df["row_index"].max())
    if max_row_index >= len(mm_index):
        raise ValueError(
            f"row_index out of memmap range: max={max_row_index}, memmap_rows={len(mm_index)}"
        )

    train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
    val_df = df.loc[df["split"] == "val"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Train or val split is empty")

    train_rows = train_df["row_index"].to_numpy(dtype=np.int64)
    val_rows = val_df["row_index"].to_numpy(dtype=np.int64)

    x_train = np.asarray(mm[train_rows], dtype=np.uint8)
    x_val = np.asarray(mm[val_rows], dtype=np.uint8)

    y_train, m_train = build_targets(train_df)
    y_val, m_val = build_targets(val_df)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(m_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
        torch.from_numpy(m_val),
    )

    pin_memory = DEVICE == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = ConvBaseline().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_ENABLED and DEVICE == "cuda"))

    with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
        write_log_line(log_handle, f"device\t{DEVICE}")
        write_log_line(log_handle, f"seed\t{SEED}")
        write_log_line(log_handle, f"seq_len\t{SEQ_LEN}")
        write_log_line(log_handle, f"train_rows\t{len(train_df)}")
        write_log_line(log_handle, f"val_rows\t{len(val_df)}")
        write_log_line(log_handle, f"batch_size\t{BATCH_SIZE}")
        write_log_line(log_handle, f"amp_enabled\t{AMP_ENABLED and DEVICE == 'cuda'}")
        write_log_line(log_handle, f"memmap_path\t{MEMMAP_PATH}")
        write_log_line(log_handle, f"memmap_index_path\t{MEMMAP_INDEX_PATH}")

        best_val = None
        history_rows = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            t0 = time.time()
            train_loss_sum = 0.0
            train_batches = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(AMP_ENABLED and DEVICE == "cuda")):
                    pred = model(xb)
                    loss = masked_mse(pred, yb, mb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += float(loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)
            val_metrics = evaluate(model, val_loader)
            val_key = np.nanmean([
                val_metrics["Score_E_mse"],
                val_metrics["Score_S_mse"],
                val_metrics["Score_B_mse"],
            ])
            elapsed = time.time() - t0

            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mean_mse": float(val_key),
                **val_metrics,
                "epoch_sec": elapsed,
            }
            history_rows.append(history_row)

            write_log_line(
                log_handle,
                (
                    f"epoch\t{epoch}\ttrain_loss\t{train_loss:.6f}"
                    f"\tval_mean_mse\t{val_key:.6f}"
                    f"\tE_r2\t{val_metrics['Score_E_r2']:.4f}"
                    f"\tS_r2\t{val_metrics['Score_S_r2']:.4f}"
                    f"\tB_r2\t{val_metrics['Score_B_r2']:.4f}"
                    f"\tsec\t{elapsed:.2f}"
                ),
            )

            if best_val is None or val_key < best_val:
                best_val = val_key
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "seq_len": SEQ_LEN,
                        "epoch": epoch,
                        "best_val_mean_mse": best_val,
                        "config": {
                            "batch_size": BATCH_SIZE,
                            "epochs": EPOCHS,
                            "lr": LR,
                            "weight_decay": WEIGHT_DECAY,
                            "amp_enabled": AMP_ENABLED and DEVICE == "cuda",
                        },
                    },
                    CHECKPOINT_PATH,
                )
                write_log_line(log_handle, f"checkpoint_saved\t{CHECKPOINT_PATH}")

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(EVAL_PATH, sep="\t", index=False)

    with open(CARD_PATH, "w", encoding="utf-8") as handle:
        handle.write("model_name\tpromoter_model_v1\n")
        handle.write("phase\tPhase 6 baseline promoter model\n")
        handle.write("input\tsequence_only_core_memmap\n")
        handle.write("targets\tScore_E,Score_S,Score_B\n")
        handle.write("score_s_missing_policy\tmasked_loss\n")
        handle.write("split_key\trecord_id\n")
        handle.write("sequence_asset\tpromoters/memmap/promoter_up200_uint8.memmap\n")
        handle.write("sequence_index\tpromoters/memmap/promoter_up200_index.tsv\n")
        handle.write("dataset\tlabels/build/promoter_model_input_core.tsv + labels/releases/promoter_split_core.tsv\n")
        handle.write("checkpoint\tmodels/checkpoints/promoter_model_v1.pt\n")
        handle.write("eval_table\tmodels/eval/promoter_model_v1_metrics.tsv\n")
        handle.write("log_file\tmodels/logs/train_promoter_model_v1.log\n")

    print("Training finished.")
    print(f"log: {LOG_PATH}")
    print(f"eval: {EVAL_PATH}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"card: {CARD_PATH}")


if __name__ == "__main__":
    main()
