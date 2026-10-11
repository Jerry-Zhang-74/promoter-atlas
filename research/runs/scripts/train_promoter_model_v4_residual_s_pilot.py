"""
Phase 6 architecture residual-S pilot for promoter_atlas.

This script trains an architecture-only pilot model with:
- direct prediction for Score_E and Score_B
- residual prediction for Score_S after subtracting linear E effect

Outputs
-------
- models/checkpoints/promoter_model_v4_residual_s_pilot.pt
- models/eval/promoter_model_v4_residual_s_pilot_metrics.tsv
- models/eval/promoter_model_v4_residual_s_pilot_best_summary.tsv
- models/logs/train_promoter_model_v4_residual_s_pilot.log
- models/cards/promoter_model_v4_residual_s_pilot_card.txt
"""

from pathlib import Path
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


ROOT = Path(".")

INPUT_TABLE = ROOT / "labels/build/promoter_model_input_core.tsv"
SPLIT_TABLE = ROOT / "labels/releases/promoter_split_core.tsv"
ARCH_PATH = ROOT / "features/sequence/promoter_architecture_features_v2.tsv"

CHECKPOINT_DIR = ROOT / "models/checkpoints"
EVAL_DIR = ROOT / "models/eval"
LOG_DIR = ROOT / "models/logs"
CARD_DIR = ROOT / "models/cards"

CHECKPOINT_PATH = CHECKPOINT_DIR / "promoter_model_v4_residual_s_pilot.pt"
EVAL_PATH = EVAL_DIR / "promoter_model_v4_residual_s_pilot_metrics.tsv"
BEST_SUMMARY_PATH = EVAL_DIR / "promoter_model_v4_residual_s_pilot_best_summary.tsv"
LOG_PATH = LOG_DIR / "train_promoter_model_v4_residual_s_pilot.log"
CARD_PATH = CARD_DIR / "promoter_model_v4_residual_s_pilot_card.txt"

SEED = 20260315
TRAIN_N = 20000
VAL_N = 4000
TEST_N = 4000

BATCH_SIZE = 1024
EPOCHS = 140
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


def fit_linear_s_from_e(e_values, s_values):
    e = np.asarray(e_values, dtype=np.float64)
    s = np.asarray(s_values, dtype=np.float64)
    mask = np.isfinite(e) & np.isfinite(s)
    e = e[mask]
    s = s[mask]
    if len(e) < 2:
        return 0.0, float(np.nanmean(s_values))
    e_mean = e.mean()
    s_mean = s.mean()
    var_e = ((e - e_mean) ** 2).sum()
    if var_e == 0:
        return 0.0, float(s_mean)
    cov = ((e - e_mean) * (s - s_mean)).sum()
    a = cov / var_e
    b = s_mean - a * e_mean
    return float(a), float(b)


def write_log_line(handle, text):
    print(text)
    handle.write(text + "\n")
    handle.flush()


def load_main_dataframe():
    core = pd.read_csv(INPUT_TABLE, sep="\t", low_memory=False)
    split = pd.read_csv(SPLIT_TABLE, sep="\t", low_memory=False)
    arch = pd.read_csv(ARCH_PATH, sep="\t", low_memory=False)

    req_core = ["record_id", "gene_id_base", "Score_E", "Score_S", "Score_B"]
    req_split = ["record_id", "split"]
    req_arch = ["record_id", "gene_id_base"]

    miss_core = [c for c in req_core if c not in core.columns]
    miss_split = [c for c in req_split if c not in split.columns]
    miss_arch = [c for c in req_arch if c not in arch.columns]

    if miss_core:
        raise KeyError(f"Missing core columns: {miss_core}")
    if miss_split:
        raise KeyError(f"Missing split columns: {miss_split}")
    if miss_arch:
        raise KeyError(f"Missing architecture columns: {miss_arch}")

    arch_cols = [c for c in arch.columns if c not in {"record_id", "gene_id_base"}]
    if len(arch_cols) == 0:
        raise ValueError("No architecture v2 feature columns found")

    non_numeric = [c for c in arch_cols if not pd.api.types.is_numeric_dtype(arch[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric architecture v2 columns found: {non_numeric}")

    df = core.merge(split[["record_id", "split"]], on="record_id", how="inner")
    df = df.merge(
        arch[["record_id", "gene_id_base"] + arch_cols],
        on=["record_id", "gene_id_base"],
        how="inner",
    )

    if df["record_id"].duplicated().any():
        dup_n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"Duplicated record_id after merge: {dup_n}")

    return df, arch_cols


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


def build_targets(df, s_resid_a, s_resid_b):
    y = np.zeros((len(df), 3), dtype=np.float32)
    m = np.ones((len(df), 3), dtype=np.float32)

    e = pd.to_numeric(df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    s = pd.to_numeric(df["Score_S"], errors="coerce")
    b = pd.to_numeric(df["Score_B"], errors="coerce").to_numpy(dtype=np.float32)

    s_mask = s.notna().to_numpy()
    s_full = s.fillna(0.0).to_numpy(dtype=np.float32)
    s_resid = s_full - (s_resid_a * e + s_resid_b)

    y[:, 0] = e
    y[:, 1] = s_resid.astype(np.float32)
    y[:, 2] = b

    m[~s_mask, 1] = 0.0
    return y, m


class ResidualSModel(nn.Module):
    def __init__(self, arch_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(arch_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.head_e = nn.Linear(32, 1)
        self.head_s_resid = nn.Linear(32, 1)
        self.head_b = nn.Linear(32, 1)

    def forward(self, arch_x):
        z = self.backbone(arch_x)
        e = self.head_e(z)
        s_resid = self.head_s_resid(z)
        b = self.head_b(z)
        return torch.cat([e, s_resid, b], dim=1)


def evaluate(model, loader, split_name, s_resid_a, s_resid_b):
    model.eval()

    pred_e_all = []
    pred_s_all = []
    pred_b_all = []
    y_e_all = []
    y_s_resid_all = []
    y_b_all = []
    m_all = []

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            mb = mb.to(DEVICE, non_blocking=True)

            pred = model(xb)
            pred_e = pred[:, 0]
            pred_s = s_resid_a * pred_e + s_resid_b + pred[:, 1]
            pred_b = pred[:, 2]

            pred_e_all.append(pred_e.cpu())
            pred_s_all.append(pred_s.cpu())
            pred_b_all.append(pred_b.cpu())

            y_e_all.append(yb[:, 0].cpu())
            y_s_resid_all.append(yb[:, 1].cpu())
            y_b_all.append(yb[:, 2].cpu())
            m_all.append(mb.cpu())

    pred_e_all = torch.cat(pred_e_all, dim=0)
    pred_s_all = torch.cat(pred_s_all, dim=0)
    pred_b_all = torch.cat(pred_b_all, dim=0)

    y_e_all = torch.cat(y_e_all, dim=0)
    y_s_resid_all = torch.cat(y_s_resid_all, dim=0)
    y_b_all = torch.cat(y_b_all, dim=0)
    m_all = torch.cat(m_all, dim=0)

    y_s_all = s_resid_a * y_e_all + s_resid_b + y_s_resid_all

    metrics = {"split": split_name}

    for name, pred_vec, true_vec, mask_vec in [
        ("Score_E", pred_e_all, y_e_all, m_all[:, 0]),
        ("Score_S", pred_s_all, y_s_all, m_all[:, 1]),
        ("Score_B", pred_b_all, y_b_all, m_all[:, 2]),
    ]:
        mask = mask_vec > 0
        yp = pred_vec[mask]
        yt = true_vec[mask]

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

    pred_e_np = pred_e_all.numpy()
    pred_s_np = pred_s_all.numpy()

    s_mask = m_all[:, 1].numpy() > 0
    true_s_np = y_s_all.numpy()[s_mask]
    true_e_np = y_e_all.numpy()[s_mask]
    pred_s_masked = pred_s_np[s_mask]
    pred_e_masked = pred_e_np[s_mask]

    metrics["pred_E_pred_S_corr_all"] = corr_np(pred_e_np, pred_s_np)
    metrics["true_E_true_S_corr_defined"] = corr_np(true_e_np, true_s_np)
    metrics["pred_E_pred_S_corr_defined"] = corr_np(pred_e_masked, pred_s_masked)

    return metrics


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)

    df, arch_cols = load_main_dataframe()
    df = sample_splits(df)

    train_df = df.loc[df["split"] == "train"].reset_index(drop=True)
    val_df = df.loc[df["split"] == "val"].reset_index(drop=True)
    test_df = df.loc[df["split"] == "test"].reset_index(drop=True)

    e_train = pd.to_numeric(train_df["Score_E"], errors="coerce").to_numpy(dtype=np.float32)
    s_train = pd.to_numeric(train_df["Score_S"], errors="coerce").to_numpy(dtype=np.float32)
    s_resid_a, s_resid_b = fit_linear_s_from_e(e_train, s_train)

    x_arch_train_raw = train_df[arch_cols].to_numpy(dtype=np.float32)
    arch_mean, arch_std = fit_standardizer(x_arch_train_raw)

    def build_split_arrays(split_df):
        arch_x = transform_standardize(
            split_df[arch_cols].to_numpy(dtype=np.float32),
            arch_mean,
            arch_std,
        )
        y, m = build_targets(split_df, s_resid_a, s_resid_b)
        return arch_x, y, m

    arch_train, y_train, m_train = build_split_arrays(train_df)
    arch_val, y_val, m_val = build_split_arrays(val_df)
    arch_test, y_test, m_test = build_split_arrays(test_df)

    train_ds = TensorDataset(
        torch.from_numpy(arch_train),
        torch.from_numpy(y_train),
        torch.from_numpy(m_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(arch_val),
        torch.from_numpy(y_val),
        torch.from_numpy(m_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(arch_test),
        torch.from_numpy(y_test),
        torch.from_numpy(m_test),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    model = ResidualSModel(arch_dim=len(arch_cols)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP_ENABLED and DEVICE == "cuda"))

    best_val = None
    best_epoch = None
    best_val_metrics = None
    history_rows = []

    with open(LOG_PATH, "w", encoding="utf-8") as log_handle:
        write_log_line(log_handle, f"device\t{DEVICE}")
        write_log_line(log_handle, f"seed\t{SEED}")
        write_log_line(log_handle, f"arch_feature_count\t{len(arch_cols)}")
        write_log_line(log_handle, f"train_rows\t{len(train_df)}")
        write_log_line(log_handle, f"val_rows\t{len(val_df)}")
        write_log_line(log_handle, f"test_rows\t{len(test_df)}")
        write_log_line(log_handle, f"batch_size\t{BATCH_SIZE}")
        write_log_line(log_handle, f"amp_enabled\t{AMP_ENABLED and DEVICE == 'cuda'}")
        write_log_line(log_handle, f"s_resid_linear_a\t{s_resid_a:.6f}")
        write_log_line(log_handle, f"s_resid_linear_b\t{s_resid_b:.6f}")

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
            val_metrics = evaluate(model, val_loader, "val", s_resid_a, s_resid_b)
            val_key = float(np.nanmean([
                val_metrics["Score_E_mse"],
                val_metrics["Score_S_mse"],
                val_metrics["Score_B_mse"],
            ]))
            elapsed = time.time() - t0

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_mean_mse": float(val_key),
                **val_metrics,
                "epoch_sec": float(elapsed),
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
                best_val = float(val_key)
                best_epoch = int(epoch)
                best_val_metrics = dict(val_metrics)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "arch_cols": arch_cols,
                        "arch_mean": arch_mean.tolist(),
                        "arch_std": arch_std.tolist(),
                        "epoch": best_epoch,
                        "best_val_mean_mse": best_val,
                        "best_val_metrics": best_val_metrics,
                        "s_resid_linear_a": s_resid_a,
                        "s_resid_linear_b": s_resid_b,
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

                write_log_line(
                    log_handle,
                    (
                        f"checkpoint_saved\t{CHECKPOINT_PATH}"
                        f"\tbest_epoch_by_val_mean_mse\t{best_epoch}"
                        f"\tbest_val_mean_mse\t{best_val:.6f}"
                    )
                )

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(EVAL_PATH, sep="\t", index=False)

        best_state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])

        val_best_metrics = evaluate(
            model, val_loader, "val_best",
            best_state["s_resid_linear_a"],
            best_state["s_resid_linear_b"],
        )
        test_best_metrics = evaluate(
            model, test_loader, "test_best",
            best_state["s_resid_linear_a"],
            best_state["s_resid_linear_b"],
        )

        extra_df = pd.DataFrame(
            [
                {"epoch": "best_val_reload", **val_best_metrics},
                {"epoch": "best_test", **test_best_metrics},
            ]
        )
        final_df = pd.concat([history_df, extra_df], ignore_index=True)
        final_df.to_csv(EVAL_PATH, sep="\t", index=False)

        best_summary = pd.DataFrame(
            [
                {
                    "selection_rule": "lowest_val_mean_mse",
                    "best_epoch": best_epoch,
                    "best_val_mean_mse": best_val,
                    "best_val_Score_E_r2": best_val_metrics["Score_E_r2"],
                    "best_val_Score_S_r2": best_val_metrics["Score_S_r2"],
                    "best_val_Score_B_r2": best_val_metrics["Score_B_r2"],
                    "best_val_predEScorr": best_val_metrics["pred_E_pred_S_corr_defined"],
                    "best_test_Score_E_r2": test_best_metrics["Score_E_r2"],
                    "best_test_Score_S_r2": test_best_metrics["Score_S_r2"],
                    "best_test_Score_B_r2": test_best_metrics["Score_B_r2"],
                    "best_test_predEScorr": test_best_metrics["pred_E_pred_S_corr_defined"],
                }
            ]
        )
        best_summary.to_csv(BEST_SUMMARY_PATH, sep="\t", index=False)

        write_log_line(
            log_handle,
            (
                f"best_epoch_by_val_mean_mse\t{best_epoch}"
                f"\tbest_val_mean_mse\t{best_val:.6f}"
            )
        )
        write_log_line(
            log_handle,
            (
                f"best_val_metrics"
                f"\tE_r2\t{best_val_metrics['Score_E_r2']:.4f}"
                f"\tS_r2\t{best_val_metrics['Score_S_r2']:.4f}"
                f"\tB_r2\t{best_val_metrics['Score_B_r2']:.4f}"
                f"\tpredEScorr\t{best_val_metrics['pred_E_pred_S_corr_defined']:.4f}"
            )
        )
        write_log_line(
            log_handle,
            (
                f"best_test_metrics"
                f"\tE_r2\t{test_best_metrics['Score_E_r2']:.4f}"
                f"\tS_r2\t{test_best_metrics['Score_S_r2']:.4f}"
                f"\tB_r2\t{test_best_metrics['Score_B_r2']:.4f}"
                f"\tpredEScorr\t{test_best_metrics['pred_E_pred_S_corr_defined']:.4f}"
            )
        )

    with open(CARD_PATH, "w", encoding="utf-8") as handle:
        handle.write("model_name\tpromoter_model_v4_residual_s_pilot\n")
        handle.write("phase\tPhase 6 architecture residual-S route\n")
        handle.write("input\tarchitecture_v2_only\n")
        handle.write("arch_asset\tfeatures/sequence/promoter_architecture_features_v2.tsv\n")
        handle.write("targets\tScore_E,Score_S,Score_B\n")
        handle.write("Score_S_target_type\tresidual_after_linear_E_effect\n")
        handle.write(f"s_resid_linear_a\t{s_resid_a:.6f}\n")
        handle.write(f"s_resid_linear_b\t{s_resid_b:.6f}\n")
        handle.write("split_key\trecord_id\n")
        handle.write(f"train_n\t{TRAIN_N}\n")
        handle.write(f"val_n\t{VAL_N}\n")
        handle.write(f"test_n\t{TEST_N}\n")
        handle.write(f"arch_feature_count\t{len(arch_cols)}\n")
        handle.write(f"best_epoch_by_val_mean_mse\t{best_epoch}\n")
        handle.write(f"best_val_mean_mse\t{best_val:.6f}\n")
        handle.write("checkpoint\tmodels/checkpoints/promoter_model_v4_residual_s_pilot.pt\n")
        handle.write("eval_table\tmodels/eval/promoter_model_v4_residual_s_pilot_metrics.tsv\n")
        handle.write("best_summary_table\tmodels/eval/promoter_model_v4_residual_s_pilot_best_summary.tsv\n")
        handle.write("log_file\tmodels/logs/train_promoter_model_v4_residual_s_pilot.log\n")

    print("v4 residual-S pilot training finished")
    print(f"log: {LOG_PATH}")
    print(f"eval: {EVAL_PATH}")
    print(f"best_summary: {BEST_SUMMARY_PATH}")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"card: {CARD_PATH}")


if __name__ == "__main__":
    main()
