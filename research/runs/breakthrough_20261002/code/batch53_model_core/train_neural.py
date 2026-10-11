# -*- coding: utf-8 -*-
"""Training engine for the two fresh neural arms (Nabs absolute, Njoint pair-augmented).

Both arms share architecture, init bytes, optimizer, seed and config (spec.py). Njoint adds
ONLY the 0.5 * group-weighted SmoothL1 pair-difference term. No work at import time. An
actual fit REQUIRES: (a) owner opt-in flag, (b) accepted actual-data sha, and (c) a reserved
exec_gate slot BEFORE the first gradient. The batch53 helper never runs a real fit.
"""
from __future__ import annotations
import os, time

import torch

import losses
from checkpoint import save_checkpoint
from metrics import group_mean_pair_mae, group_equal_mae, rmse, spearman, macro_sign_accuracy
from model_core import build_model, param_sha256, assert_cuda_all
from normalizer import TargetNormalizer
from spec import (SEED, LR, WEIGHT_DECAY, GRAD_CLIP_NORM, BATCH_SIZE, PAIR_MICROBATCH,
                  MAX_EPOCHS, config_sha256)


def compute_gc(codes: torch.Tensor) -> torch.Tensor:
    valid = (codes < 4).sum(dim=1).clamp(min=1).to(torch.float32)
    gc = ((codes == 1) | (codes == 2)).sum(dim=1).to(torch.float32)
    return gc / valid


def epoch_order(n: int, epoch: int, seed: int = SEED):
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    return torch.randperm(n, generator=g)


class Trainer:
    def __init__(self, arm, cohort, out_dir, ledger_path, device="cuda:0",
                 epoch_cap=None, micro_batch=BATCH_SIZE, opt_in=False, accepted_data_sha=None,
                 resume_from=None, resume_identity=None):
        if arm not in ("Nabs", "Njoint"):
            raise ValueError("arm must be Nabs or Njoint, got %r" % arm)
        self.arm = arm
        self.cohort = cohort
        self.out_dir = out_dir
        self.ledger_path = ledger_path
        self.device = device
        self.epoch_cap = int(epoch_cap or MAX_EPOCHS)
        self.epoch_cap = max(1, min(self.epoch_cap, MAX_EPOCHS))
        self.micro_batch = int(micro_batch)
        if self.micro_batch <= 0 or BATCH_SIZE % self.micro_batch != 0:
            raise ValueError("micro_batch must divide effective batch %d" % BATCH_SIZE)
        self.accum = max(1, BATCH_SIZE // self.micro_batch)
        self.opt_in = bool(opt_in)
        self.accepted_data_sha = accepted_data_sha
        self.resume_from = resume_from
        self.resume_identity = resume_identity
        os.makedirs(self.out_dir, exist_ok=True)

    def _reserve(self):
        if not self.opt_in:
            raise PermissionError("real fit refused: pass --i-understand-this-is-a-real-fit")
        if not self.accepted_data_sha or self.accepted_data_sha != self.cohort.source_sha256:
            raise PermissionError("real fit refused: accepted actual-data sha required (owner opt-in)")
        from exec_gate import open_ledger
        gate = open_ledger(self.ledger_path)
        st = gate.status()
        if st["remaining"] <= 0:
            raise RuntimeError("exec_gate budget exhausted: %r" % st)
        token = gate.reserve(self.arm, resume_of=self.resume_from, resume_identity=self.resume_identity)
        return gate, token

    def identity(self) -> str:
        return "b53|%s|%s|data=%s|seed=%d" % (
            self.arm, config_sha256()[:12], (self.cohort.datahash or "")[:12], SEED)

    def _setup(self):
        model = build_model(SEED).to(self.device)
        tr, wtr = self.cohort.activity_weights("TRAIN")
        self.tr_idx = tr
        codes_tr = self.cohort.codes[tr].to(self.device)
        gc_tr = compute_gc(codes_tr.cpu())
        model.pool_stats.fit(self.cohort.lengths[tr], gc_tr)
        self.norm = TargetNormalizer.fit(self.cohort.targets[tr], wtr)
        self.tr_codes = codes_tr
        self.tr_yz = self.norm.forward(self.cohort.targets[tr]).to(self.device)
        self.tr_w = wtr.to(self.device)
        if self.arm == "Njoint":
            pt, pw = self.cohort.pair_weights("TRAIN")
            if pt is None or pt.numel() == 0:
                raise RuntimeError("Njoint needs TRAIN pairs")
            pos = {cid: k for k, cid in enumerate(tr.tolist())}
            self.pair_pos = torch.tensor([[pos[int(a)], pos[int(b)]] for a, b in pt], dtype=torch.long)
            self.pair_w = pw.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        return model, opt

    @torch.no_grad()
    def _eval_split(self, model, split):
        model.eval()
        idx = self.cohort.split_indices(split)
        preds = []
        for s in range(0, len(idx), 128):
            preds.append(self.norm.inverse(model(self.cohort.codes[idx[s:s + 128]].to(self.device))).cpu())
        pred = torch.cat(preds) if preds else torch.zeros(0)
        y = self.cohort.targets[idx]
        groups = [self.cohort.group_ids[i] for i in idx.tolist()]
        act = {"mae": group_equal_mae(y.numpy(), pred.numpy(), groups),
               "rmse": rmse(y.numpy(), pred.numpy()), "rho": spearman(y.numpy(), pred.numpy())}
        pair_mae, sign = float("nan"), float("nan")
        pr = self.cohort.pairs.get(split)
        if pr is not None and pr.numel() > 0:
            pos = {int(g): k for k, g in enumerate(idx.tolist())}
            short = pred[[pos[int(a)] for a in pr[:, 0]]]
            long_ = pred[[pos[int(b)] for b in pr[:, 1]]]
            dt = (self.cohort.targets[pr[:, 0]] - self.cohort.targets[pr[:, 1]]).numpy()
            dp = (short - long_).numpy()
            pg = [self.cohort.group_ids[int(a)] for a in pr[:, 0]]
            pair_mae = group_mean_pair_mae(dt, dp, pg)
            sign = macro_sign_accuracy(dt, dp, pg)
        model.train()
        return act, pair_mae, sign

    def run(self, max_seconds=None, cap_decider=None, cap_log_path=None):
        gate, token = self._reserve()
        t_start = time.time()
        model, opt = self._setup()
        init_sha = param_sha256(model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        n = len(self.tr_idx)
        pid = 0
        best = {"pair_mae": float("inf"), "epoch": -1, "unset": True}
        history = []
        last_save = time.time()
        try:
            for epoch in range(self.epoch_cap):
                epoch_start = time.time()
                perm = epoch_order(n, epoch).tolist()
                epoch_loss, nmicro = 0.0, 0
                for bi in range(0, n, BATCH_SIZE):
                    pos_slice = perm[bi:bi + BATCH_SIZE]
                    opt.zero_grad(set_to_none=True)
                    for mb in range(0, len(pos_slice), self.micro_batch):
                        mp = pos_slice[mb:mb + self.micro_batch]
                        mcodes = self.tr_codes[torch.tensor(mp, dtype=torch.long)]
                        my = self.tr_yz[torch.tensor(mp, dtype=torch.long)]
                        mw = self.tr_w[torch.tensor(mp, dtype=torch.long)]
                        assert_cuda_all(model, mcodes)
                        out = model(mcodes)
                        if self.arm == "Njoint":
                            m = min(PAIR_MICROBATCH, self.pair_pos.shape[0])
                            sel = torch.tensor([(pid + k) % self.pair_pos.shape[0] for k in range(m)], dtype=torch.long)
                            pid = (pid + m) % self.pair_pos.shape[0]
                            pp = self.pair_pos[sel]
                            ps = model(self.tr_codes[pp[:, 0]])
                            pl = model(self.tr_codes[pp[:, 1]])
                            ysh = self.tr_yz[pp[:, 0]]; ylo = self.tr_yz[pp[:, 1]]
                            loss, parts = losses.total_loss(out, my, mw, pair_weight=self.pair_w[sel],
                                                            pair_pred=(ps, pl), pair_true=(ysh, ylo), joint=True)
                        else:
                            loss, parts = losses.total_loss(out, my, mw)
                        (loss * (len(mp) / BATCH_SIZE)).backward()
                        epoch_loss += float(loss.detach()) * len(mp); nmicro += len(mp)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    opt.step()
                    if time.time() - last_save > 300:
                        save_checkpoint(os.path.join(self.out_dir, "%s_mid.ckpt" % self.arm), model, opt,
                                        epoch=epoch, global_step=nmicro, datahash=self.cohort.datahash,
                                        attempt_id=token["exec_id"], normalizer_state=self.norm.state(),
                                        sampler_state={"epoch": epoch, "pos": bi, "pair_ptr": pid},
                                        extra={"mid": True, "arm": self.arm})
                        last_save = time.time()
                act, pair_mae, sign = self._eval_split(model, "DEV")
                if pair_mae != pair_mae:
                    pair_mae = float("inf")
                if best["unset"] or pair_mae < best["pair_mae"]:
                    best = {"pair_mae": pair_mae, "epoch": epoch, "unset": False, "act_mae": act["mae"]}
                ck = save_checkpoint(os.path.join(self.out_dir, "%s_epoch%02d.ckpt" % (self.arm, epoch)),
                                     model, opt, epoch=epoch, global_step=nmicro, datahash=self.cohort.datahash,
                                     attempt_id=token["exec_id"], normalizer_state=self.norm.state(),
                                     sampler_state={"epoch": epoch, "pos": n, "pair_ptr": pid},
                                     extra={"arm": self.arm, "train_loss": epoch_loss / max(nmicro, 1),
                                            "dev": act, "dev_pair_mae": pair_mae, "dev_sign": sign,
                                            "init_sha": init_sha, "epoch_cap": self.epoch_cap})
                history.append({"epoch": epoch, "loss": epoch_loss / max(nmicro, 1), "dev": act,
                                "dev_pair_mae": pair_mae, "dev_sign": sign, "ckpt": ck})
                if epoch == 0 and cap_decider is not None and self.arm == "Nabs":
                    epoch_seconds = time.time() - epoch_start
                    decision = cap_decider(epoch_seconds)
                    new_cap = decision.get("chosen_cap")
                    if new_cap is None:
                        self.epoch_cap = 1       # walltime risk: keep checkpoints, stop honestly
                        if cap_log_path:
                            from epoch_cap import log_epoch_cap
                            log_epoch_cap(cap_log_path, decision, epoch_seconds)
                    else:
                        self.epoch_cap = int(new_cap)
                        if cap_log_path:
                            from epoch_cap import log_epoch_cap
                            log_epoch_cap(cap_log_path, decision, epoch_seconds)
                if max_seconds is not None and (time.time() - t_start) > max_seconds:
                    break
            sel_src = os.path.join(self.out_dir, "%s_epoch%02d.ckpt" % (self.arm, max(best["epoch"], 0)))
            sel_dst = os.path.join(self.out_dir, "%s_selected.ckpt" % self.arm)
            if os.path.exists(sel_src):
                obj = torch.load(sel_src, map_location="cpu", weights_only=False)
                tmp = sel_dst + ".tmp"
                torch.save(obj, tmp); os.replace(tmp, sel_dst)
            gate.complete(token, identity=self.identity())
            return {"arm": self.arm, "init_sha": init_sha, "epoch_cap": self.epoch_cap, "best": best,
                    "history": history, "selected": sel_dst, "elapsed_s": time.time() - t_start,
                    "attempt_id": token["exec_id"],
                    "peak_cuda_mb": (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0}
        except BaseException as e:
            gate.fail(token, e)
            raise


def main(argv=None):
    import argparse, json
    from data_adapter import load_cohort
    from spec import SOURCE_SHA256
    ap = argparse.ArgumentParser(description="Train ONE fresh neural arm (Nabs or Njoint).")
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--source-sha", default=SOURCE_SHA256)
    ap.add_argument("--arm", required=True, choices=["Nabs", "Njoint"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ledger", required=True, help="batch52 exec_gate ledger path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epoch-cap", type=int, default=MAX_EPOCHS)
    ap.add_argument("--micro-batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--resume-identity", default=None)
    ap.add_argument("--accepted-data-sha", default=None)
    ap.add_argument("--i-understand-this-is-a-real-fit", action="store_true", dest="opt_in")
    a = ap.parse_args(argv)
    cohort = load_cohort(a.cohort_dir, a.source_sha)
    tr = Trainer(a.arm, cohort, a.out_dir, a.ledger, device=a.device, epoch_cap=a.epoch_cap,
                 micro_batch=a.micro_batch, opt_in=a.opt_in, accepted_data_sha=a.accepted_data_sha,
                 resume_from=a.resume_from, resume_identity=a.resume_identity)
    res = tr.run(max_seconds=a.max_seconds)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
