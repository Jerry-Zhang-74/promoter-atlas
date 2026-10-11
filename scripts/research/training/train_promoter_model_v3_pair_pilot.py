"""
Phase 6 grammar-lite pair pilot for promoter_atlas.

This script trains a pilot multi-input promoter model using:
- sequence memmap (core up200)
- pair grammar-lite features

Outputs
-------
- models/checkpoints/promoter_model_v3_pair_pilot.pt
- models/eval/promoter_model_v3_pair_pilot_metrics.tsv
- models/logs/train_promoter_model_v3_pair_pilot.log
- models/cards/promoter_model_v3_pair_pilot_card.txt
"""

from pathlib import Path
import math
import random
import time

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
PAIR_FEATURE_PATH = ROOT / "features/motif/promoter_motif_pair_features_v1.tsv"

CHECKPOINT_DIR = ROOT / "models/checkpoints"
EVAL_DIR = ROOT / "models/eval"
LOG_DIR = ROOT / "models/logs"
CARD_DIR = ROOT / "models/cards"

CHECKPOINT_PATH = CHECKPOINT_DIR / "promoter_model_v3_pair_pilot.pt"
EVAL_PATH = EVAL_DIR / "promoter_model_v3_pair_pilot_metrics.tsv"
LOG_PATH = LOG_DIR / "train_promoter_model_v3_pair_pilot.log"
CARD_PATH = CARD_DIR / "promoter_model_v3_pair_pilot_card.txt"

SEED = 20260315
SEQ_LEN = 200

TRAIN_N = 20000
VAL_N = 4000
TEST_N = 4000

BATCH_SIZE = 2048
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
AMP_ENABLED = True
NUM_WORKERS = 0

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


def corr_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def write_log_line(handle, text):
    print(text)
    handle.write(text + "\n")
    handle.flush()


def load_main_dataframe():
    core = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    pair = pd.read_csv(PAIR_FEATURE_PATH, sep="\t", low_memory=False)

    req_core = ["record_id", "gene_id_base", "Score_E", "Score_S", "Score_B"]
    req_split = ["record_id", "split"]
    req_mm = ["record_id", "row_index"]
    req_pair = ["record_id", "gene_id_base"]

    miss_core = [c for c in req_core if c not in core.columns]
    miss_split = [c for c in req_split if c not in split.columns]
    miss_mm = [c for c in req_mm if c not in mm_index.columns]
    miss_pair = [c for c in req_pair if c not in pair.columns]

    if miss_core:
        raise KeyError(f"Missing core columns: {miss_core}")
    if miss_split:
        raise KeyError(f"Missing split columns: {miss_split}")
    if miss_mm:
        raise KeyError(f"Missing memmap index columns: {miss_mm}")
    if miss_pair:
        raise KeyError(f"Missing pair feature columns: {miss_pair}")

    pair_cols = [c for c in pair.columns if c not in {"record_id", "gene_id_base"}]
    if len(pair_cols) == 0:
        raise ValueError("No pair feature columns found")

    non_numeric = [c for c in pair_cols if not pd.api.types.is_numeric_dtype(pair[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric pair feature columns found: {non_numeric}")

    df = core.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(mm_index[["record_id", "row_index"]], on="record_id", how="inner")
    df = df.merge(
        pair[["record_id", "gene_id_base"] + pair_cols],
        on=["record_id", "gene_id_base"],
        how="inner",
    )

    if df["record_id"].duplicated().any():
        dup_n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    return df, pair_cols


def sample_splits(df):
    rng = np.random.default_rng(SEED)
    sampled = []

    for split_name, n_keep in [("train", TRAIN_N), ("val", VAL_N), ("test", TEST_N)]:
        sub = df.loc[df["split"] == split_name].copy()
        if len(sub) == 0:
            raise ValueError(f"Empty split: {split_name}")
        if len(sub) > n_keep:
            idx = rng.choice(sub.index.to_numpy(), size=n_keep, replace=False)
            sub = sub.loc[idx].copy()
        sampled.append(sub)

    out = pd.concat(sampled, ignore_index=True)
    out = out.sort_values(["split", "record_id"]).reset_index(drop=True)
    return out


def fit_standardizer(train_x):
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 1.0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def transform_standardize(x, mean, std):
    x = np.where(np.isfinite(x), x, mean)
    x = (x - mean) / std
    return x.astype(np.float32)


def build_targets(df):
    y = np.zeros((len(df), 3), dtype=np.float32)
    m = np.ones((len(df), 3), dtype=np.float32)

    y[:, 0] = pd.to_numeric(df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    y[:, 1] = pd.to_numeric(df["Score_S"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y[:, 2] = pd.to_numeric(df["Score_B"], errors="coerce").to_numpy(dtype=np.float32)

    s_missing = pd.to_numeric(df["Score_S"], errors="coerce").isna().to_numpy()
    m[s_missing, 1] = 0.0

    return y, m


def load_memmap():
    mm_index = pd.read_csv(MEMMAP_INDEX_PATH, sep="\t", low_memory=False)
    n_rows = len(mm_index)
    mm = np.memmap(
        MEMMAP_PATH,
        dtype="uint8",
        mode="r",
        shape=(n_rows, SEQ_LEN),
    )
    return mm


class SeqPairModel(nn.Module):
    def __init__(self, pair_dim):
        super().__init__()

        self.seq_features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.seq_proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.pair_proj = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        self.shared = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.head_e = nn.Linear(128, 1)
        self.head_s = nn.Linear(128, 1)
        self.head_b = nn.Linear(128, 1)

    def forward(self, seq_idx, pair_x):
        seq = F.one_hot(seq_idx.long(), num_classes=5)[..., :4].float()
        seq = seq.permute(0, 2, 1).contiguous()
        seq = self.seq_features(seq).squeeze(-1)
        seq = self.seq_proj(seq)

        pair = self.pair_proj(pair_x)

        x = torch.cat([seq, pair], dim=1)
        x = self.shared(x)

        e = self.head_e(x)
        s = self.head_s(x)
        b = self.head_b(x)
        return torch.cat([e, s, b], dim=1)


def evaluate(model, loader, split_name):
    model.eval()

    pred_all = []
    y_all = []
    m_all = []

    with torch.no_grad():
        for seqb, pairb, yb, mb in loader:
            seqb = seqb.to(DEVICE, non_blocking=True)
            pairb = pairb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            pred = model(seqb, pairb)

            pred_all.append(pred.cpu())
            y_all.append(yb.cpu())
            m_all.append(mb.cpu())

    pred_all = torch.cat(pred_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    m_all = torch.cat(m_all, dim=0)

    metrics = {"split": split_name}
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

    pred_e = pred_all[:, 0].numpy()
    pred_s = pred_all[:, 1].numpy()

    s_mask = m_all[:, 1].numpy() > 0
    true_s = y_all[:, 1].numpy()[s_mask]
    true_e_masked = y_all[:, 0].numpy()[s_mask]
    pred_s_masked = pred_s[s_mask]
    pred_e_masked = pred_e[s_mask]

    metrics["pred_E_pred_S_corr_all"] = corr_np(pred_e, pred_s)
    metrics["true_E_true_S_corr_defined"] = corr_np(true_e_masked, true_s)
    metrics["pred_E_pred_S_corr_defined"] = corr_np(pred_e_masked, pred_s_masked)

    return metrics


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)

    df, pair_cols = load_main_dataframe()
    df = sample_splits(df)
    mm = load_memmap()

    train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
    val_df = df.loc[df["split"] == "val"].reset_index(drop=True)
    test_df = df.loc[df["split"] == "test"].reset_index(drop=True)

    x_pair_train_raw = train_df[pair_cols].to_numpy(dtype=np.float32)
    pair_mean, pair_std = fit_standardizer(x_pair_train_raw)

    def build_split_arrays(split_df):
        row_idx = split_df["row_index"].to_numpy(dtype=np.int64)
        seq_x = np.asarray(mm[row_idx], dtype=np.uint8)
        pair_x = transform_standardize(
            split_df[pair_cols].to_numpy(dtype=np.float32),
            pair_mean,
            pair_std,
        )
        y, m = build_targets(split_df)
        return seq_x, pair_x, y, m

    seq_train, pair_train, y_train, m_train = build_split_arrays(train_df)
    seq_val, pair_val, y_val, m_val = build_split_arrays(val_df)
    seq_test, pair_test, y_test, m_test = build_split_arrays(test_df)

    train_ds = TensorDataset(
        torch.from_numpy(seq_train),
        torch.from_numpy(pair_train),
        torch.from_numpy(y_train),
        torch.from_numpy(m_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(seq_val),
        torch.from_numpy(pair_val),
        torch.from_numpy(y_val),
        torch.from_numpy(m_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(seq_test),
        torch.from_numpy(pair_test),
        torch.from_numpy(y_test),
        torch.from_numpy(m_test),
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
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = SeqPairModel(pair_dim=len(pair_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_ENABLED and DEVICE == "cuda"))

    best_val = None
    history_rows = []

    with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
        write_log_line(log_handle, f"device\t{DEVICE}")
        write_log_line(log_handle, f"seed\t{SEED}")
        write_log_line(log_handle, f"seq_len\t{SEQ_LEN}")
        write_log_line(log_handle, f"pair_feature_count\t{len(pair_cols)}")
        write_log_line(log_handle, f"train_rows\t{len(train_df)}")
        write_log_line(log_handle, f"val_rows\t{len(val_df)}")
        write_log_line(log_handle, f"test_rows\t{len(test_df)}")
        write_log_line(log_handle, f"batch_size\t{BATCH_SIZE}")
        write_log_line(log_handle, f"amp_enabled\t{AMP_ENABLED and DEVICE == 'cuda'}")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            t0 = time.time()
            train_loss_sum = 0.0
            train_batches = 0

            for seqb, pairb, yb, mb in train_loader:
                seqb = seqb.to(DEVICE, non_blocking=True)
                pairb = pairb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                mb = mb.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(AMP_ENABLED and DEVICE == "cuda")):
                    pred = model(seqb, pairb)
                    loss = masked_mse(pred, yb, mb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += float(loss.item())
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)
            val_metrics = evaluate(model, val_loader, "val")
            val_key = np.nanmean([
                val_metrics["Score_E_mse"],
                val_metrics["Score_S_mse"],
                val_metrics["Score_B_mse"],
            ])
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mean_mse": float(val_key),
                **val_metrics,
                "epoch_sec": elapsed,
            }
            history_rows.append(row)

            write_log_line(
                log_handle,
                (
                    f"epoch\t{epoch}"
                    f"\ttrain_loss\t{train_loss:.6f}"
                    f"\tval_mean_mse\t{val_key:.6f}"
                    f"\tE_r2\t{val_metrics['Score_E_r2']:.4f}"
                    f"\tS_r2\t{val_metrics['Score_S_r2']:.4f}"
                    f"\tB_r2\t{val_metrics['Score_B_r2']:.4f}"
                    f"\tpredEScorr\t{val_metrics['pred_E_pred_S_corr_defined']:.4f}"
                    f"\tsec\t{elapsed:.2f}"
                )
            )

            if best_val is None or val_key < best_val:
                best_val = val_key
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "pair_cols": pair_cols,
                        "pair_mean": pair_mean.tolist(),
                        "pair_std": pair_std.tolist(),
                        "epoch": epoch,
                        "best_val_mean_mse": best_val,
                        "config": {
                            "train_n": TRAIN_N,
                            "val_n": VAL_N,
                            "test_n": TEST_N,
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

        best_state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])

        val_best_metrics = evaluate(model, val_loader, "val_best")
        test_best_metrics = evaluate(model, test_loader, "test_best")

        final_df = pd.concat(
            [
                history_df,
                pd.DataFrame([{"epoch": "best_val", **val_best_metrics}]),
                pd.DataFrame([{"epoch": "best_test", **test_best_metrics}]),
            ],
            ignore_index=True,
        )
        final_df.to_csv(EVAL_PATH, sep="\t", index=False)

        write_log_line(
            log_handle,
            (
                f"best_test\tE_r2\t{test_best_metrics['Score_E_r2']:.4f}"
                f"\tS_r2\t{test_best_metrics['Score_S_r2']:.4f}"
                f"\tB_r2\t{test_best_metrics['Score_B_r2']:.4f}"
                f"\tpredEScorr\t{test_best_metrics['pred_E_pred_S_corr_defined']:.4f}"
            )
        )

    with open(CARD_PATH, "w", encoding="utf-8") as handle:
        handle.write("model_name\tpromoter_model_v3_pair_pilot\n")
        handle.write("phase\tPhase 6 grammar-lite pilot\n")
        handle.write("input\tsequence_memmap_plus_pair_features\n")
        handle.write("sequence_asset\tpromoters/memmap/promoter_up200_uint8.memmap\n")
        handle.write("sequence_index\tpromoters/memmap/promoter_up200_index.tsv\n")
        handle.write("pair_feature_asset\tfeatures/motif/promoter_motif_pair_features_v1.tsv\n")
        handle.write("targets\tScore_E,Score_S,Score_B\n")
        handle.write("score_s_missing_policy\tmasked_loss\n")
        handle.write("split_key\trecord_id\n")
        handle.write(f"train_n\t{TRAIN_N}\n")
        handle.write(f"val_n\t{VAL_N}\n")
        handle.write(f"test_n\t{TEST_N}\n")
        handle.write(f"pair_feature_count\t{len(pair_cols)}\n")
        handle.write("checkpoint\tmodels/checkpoints/promoter_model_v3_pair_pilot.pt\n")
        handle.write("eval_table\tmodels/eval/promoter_model_v3_pair_pilot_metrics.tsv\n")
        handle.write("log_file\tmodels/logs/train_promoter_model_v3_pair_pilot.log\n")

    print("v3 pair pilot training finished")
    print(f"log: {LOG_PATH}")
    print(f"eval: {EVAL_PATH}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"card: {CARD_PATH}")


if __name__ == "__main__":
    main()
