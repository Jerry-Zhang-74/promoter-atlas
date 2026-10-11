# -*- coding: utf-8 -*-
"""batch58 throughput resource-variant trainer with REAL checkpoint resume.

RESOURCE CONFIGURATION variant of the frozen batch52/53 Nabs estimator: keeps fp32, lr,
loss, architecture, optimizer, seed and the per-epoch deterministic order byte-for-byte;
changes ONLY the effective optimizer batch (fewer optimizer steps per epoch), recorded
explicitly. NOT evidence of better accuracy and NOT exact replication when eff batch changes.

REAL RESUME (fresh batch58 schema v2):
  * atomically persists model / AdamW optimizer / python+numpy+torch-CPU+CUDA RNG /
    normalized-target state / pooled-feature state / epoch / the EXACT next sample offset
    (`next_pos` = position AFTER the last processed batch) / order seed / pair pointer /
    history / best / global optimizer steps / config+data hashes / device.
  * an OLD-style mid checkpoint (b57 schema whose `pos` points at the already-processed
    batch) is REJECTED, not replayed: resume REQUIRES the v2 schema + next_pos + matching
    next/last batch id lists + coverage-prefix hash.
  * strict reject on config / architecture(init) / data / device / effective-batch mismatch.
  * snapshot (weights + full state) every <=snapshot_interval_s and at every epoch end.
  * keep the last TWO valid immutable generations; SHA computed and sidecar written BEFORE
    the `latest.json` pointer is atomically updated.
  * SIGINT/SIGTERM or stop-file => graceful exit ONLY after a fully saved optimizer boundary.
  * CUDA OOM => restore the WHOLE pre-step state + RNG, then retry the same batch with a
    smaller micro-batch; retries recorded. No CPU neural fallback, no fake GPU load.
"""
from __future__ import annotations
import hashlib, json, os, signal, time

import numpy as np
import torch

HELPER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "batch57_model_core_rc2"))
import sys
if HELPER not in sys.path:
    sys.path.insert(0, HELPER)
# corrected cohort_v2 implementation lives in the batch52 data workstream (read-only import)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DATACODE = os.path.join(_ROOT, "code", "batch52_data_v2")
if os.path.isdir(DATACODE) and DATACODE not in sys.path:
    sys.path.insert(0, DATACODE)

import losses  # noqa: E402
from checkpoint import capture_rng, restore_rng  # noqa: E402
from io_utils import sha256_file, atomic_write_json, now_iso  # noqa: E402
from metrics import group_equal_mae, rmse, spearman, group_mean_pair_mae, macro_sign_accuracy  # noqa: E402
from model_core import build_model, param_sha256, assert_cuda_all  # noqa: E402
from normalizer import TargetNormalizer  # noqa: E402
from spec import (SEED, LR, WEIGHT_DECAY, GRAD_CLIP_NORM, TARGET_COLUMN,  # noqa: E402
                  config_dict, config_sha256)
from train_neural import epoch_order, compute_gc  # noqa: E402

FORMAT = "batch58.checkpoint.v2"
LATEST = "latest.json"


class ResumeRejected(RuntimeError):
    pass


class GracefulStop(RuntimeError):
    pass


class OOMExhausted(RuntimeError):
    pass


def _expected_epoch_coverage(perm) -> str:
    h = hashlib.sha256()
    h.update(b"b58-epoch-coverage|")
    h.update(np.asarray(perm, dtype=np.int64).tobytes())
    return h.hexdigest()


class B58Trainer:
    def __init__(self, cohort, out_dir, ledger_path, device="cuda:0", effective_batch=128,
                 epoch_cap=2, micro_batch=None, opt_in=False, accepted_data_sha=None,
                 run_id="Nabs_throughput_variant", expected_init_sha=None, stopfile=None,
                 snapshot_interval_s=60.0, stop_after_opt_steps=None, stop_after_seconds=None,
                 resume_latest=None):
        self.cohort = cohort
        self.out_dir = out_dir
        self.ledger_path = ledger_path
        self.device = device
        self.eff = int(effective_batch)
        self.epoch_cap = int(epoch_cap)
        self.micro = int(micro_batch or effective_batch)
        if self.micro <= 0:
            raise ValueError("micro_batch must be positive")
        self.opt_in = bool(opt_in)
        self.accepted_data_sha = accepted_data_sha
        self.run_id = run_id
        self.expected_init_sha = expected_init_sha
        self.stopfile = stopfile
        self.snapshot_interval_s = float(snapshot_interval_s)
        self.stop_after_opt_steps = stop_after_opt_steps
        self.stop_after_seconds = stop_after_seconds
        self.resume_latest = resume_latest
        os.makedirs(self.out_dir, exist_ok=True)
        self._stop_flag = False
        self._last_gen = 0
        self.oom_retries = 0
        self.oom_events = []
        self._t0 = None
        self._perm = None
        self._perm_epoch = None

    def _reserve(self, purpose):
        if not self.opt_in:
            raise PermissionError("real fit refused: pass --i-understand-this-is-a-real-fit")
        if not self.accepted_data_sha or self.accepted_data_sha != self.cohort.source_sha256:
            raise PermissionError("real fit refused: accepted actual-data sha required")
        from b58_gate import open_ledger, reserve, actual_status
        st = actual_status(self.ledger_path)
        if st["remaining"] <= 0:
            raise RuntimeError("b58 exec_gate budget exhausted: %r" % st)
        gate = open_ledger(self.ledger_path)
        token = reserve(gate, purpose)     # CONSUMING, no free-resume loophole
        return gate, token

    def identity(self) -> str:
        return "b58|%s|eff=%d|%s|data=%s|seed=%d" % (
            self.run_id, self.eff, config_sha256()[:12], (self.cohort.datahash or "")[:12], SEED)

    def _build(self):
        model = build_model(SEED).to(self.device)
        init_sha = param_sha256(model)
        tr, wtr = self.cohort.activity_weights("TRAIN")
        codes_tr = self.cohort.codes[tr].to(self.device)
        gc_tr = compute_gc(codes_tr.cpu())
        model.pool_stats.fit(self.cohort.lengths[tr], gc_tr)
        norm = TargetNormalizer.fit(self.cohort.targets[tr], wtr)
        self.tr_idx = tr
        self.tr_codes = codes_tr
        self.tr_yz = norm.forward(self.cohort.targets[tr]).to(self.device)
        self.tr_w = wtr.to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.n = int(tr.numel())
        if self.n <= 0:
            raise RuntimeError("empty TRAIN split")
        return model, opt, norm, init_sha

    def _install_signals(self):
        def handler(signum, frame):
            self._stop_flag = True
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, handler)
            except Exception:
                pass

    def _should_stop(self):
        if self._stop_flag:
            return True
        if self.stopfile and os.path.exists(self.stopfile):
            return True
        return False

    def _perm_of(self, epoch):
        if self._perm is None or self._perm_epoch != epoch:
            self._perm = epoch_order(self.n, epoch).tolist()
            self._perm_epoch = epoch
        return self._perm

    def _perm_slice(self, epoch, start, size):
        perm = self._perm_of(epoch)
        return perm[start:start + size]

    def _prev_batch_start(self, next_pos):
        s = 0
        while s + self.eff < next_pos:
            s += self.eff
        return s

    def _gen_path(self, gen):
        return os.path.join(self.out_dir, "%s_gen%04d.ckpt" % (self.run_id, gen))

    def _save_generation(self, model, opt, norm, init_sha, epoch, next_pos, history, best,
                         coverage_hash, last_batch_ids, next_batch_ids, epoch_end=False,
                         graceful=False, attempt_id=None, dev=None, extra=None):
        gen = self._last_gen + 1
        path = self._gen_path(gen)
        payload = {
            "format": FORMAT,
            "variant": "throughput_variant",
            "arm": "Nabs",
            "run_id": self.run_id,
            "config": config_dict(),
            "config_sha256": config_sha256(),
            "datahash": self.cohort.datahash,
            "source_sha256": self.cohort.source_sha256,
            "target_column": TARGET_COLUMN,
            "device": str(self.device),
            "device_is_cuda": bool(str(self.device).startswith("cuda")),
            "effective_batch": self.eff,
            "micro_batch": self.micro,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP_NORM,
            "seed": SEED,
            "init_sha": init_sha,
            "epoch": int(epoch),
            "next_pos": int(next_pos),
            "n_train": self.n,
            "opt_steps": int(self.opt_steps),
            "global_samples": int(self.global_samples),
            "epoch_samples": int(self.epoch_samples),
            "pair_ptr": int(self.pair_ptr),
            "pair_repeats": int(self.pair_repeats),
            "history": history,
            "best": best,
            "coverage_hash": coverage_hash,
            "last_batch_ids": list(map(int, last_batch_ids)),
            "next_batch_ids": list(map(int, next_batch_ids)),
            "epoch_end": bool(epoch_end),
            "graceful_stop": bool(graceful),
            "oom_retries": int(self.oom_retries),
            "dev": dev,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "normalizer": norm.state(),
            "pool_stats": {
                "loglen_mean": float(model.pool_stats.loglen_mean.detach().cpu()),
                "loglen_std": float(model.pool_stats.loglen_std.detach().cpu()),
                "gc_mean": float(model.pool_stats.gc_mean.detach().cpu()),
                "gc_std": float(model.pool_stats.gc_std.detach().cpu())},
            "rng": capture_rng(),
            "gate_attempt_id": attempt_id,
            "created_at": now_iso(),
            "generation": gen,
        }
        if extra:
            payload.update(extra)
        tmp = path + ".tmp.%d" % os.getpid()
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        sha = sha256_file(path)
        atomic_write_json(path + ".sha256.json",
                          {"checkpoint": os.path.basename(path), "generation": gen, "sha256": sha,
                           "config_sha256": payload["config_sha256"], "datahash": payload["datahash"],
                           "init_sha": init_sha, "effective_batch": self.eff, "device": str(self.device),
                           "epoch": int(epoch), "next_pos": int(next_pos),
                           "opt_steps": int(self.opt_steps), "graceful_stop": bool(graceful)})
        atomic_write_json(os.path.join(self.out_dir, LATEST),
                          {"generation": gen, "path": os.path.abspath(path), "sha256": sha,
                           "epoch": int(epoch), "next_pos": int(next_pos),
                           "opt_steps": int(self.opt_steps), "effective_batch": self.eff,
                           "run_id": self.run_id, "updated_at": now_iso(),
                           "graceful_stop": bool(graceful)})
        self._last_gen = gen
        self._prune()
        return {"generation": gen, "path": path, "sha256": sha, "next_pos": int(next_pos)}

    def _prune(self):
        keep = {self._last_gen, self._last_gen - 1}
        for p in os.listdir(self.out_dir):
            if not p.startswith(self.run_id + "_gen") or not p.endswith(".ckpt"):
                continue
            try:
                g = int(p[len(self.run_id) + 4:-5])
            except ValueError:
                continue
            if g not in keep:
                for suffix in ("", ".sha256.json"):
                    try:
                        os.remove(os.path.join(self.out_dir, p + suffix))
                    except OSError:
                        pass

    def _load_latest(self, path):
        with open(path, "r", encoding="utf-8") as f:
            ptr = json.load(f)
        ck = ptr["path"]
        if not os.path.exists(ck):
            raise ResumeRejected("latest pointer checkpoint missing: %s" % ck)
        sha = sha256_file(ck)
        if ptr.get("sha256") and ptr["sha256"] != sha:
            raise ResumeRejected("latest checkpoint bytes changed: %s != %s" % (sha, ptr["sha256"]))
        side_p = ck + ".sha256.json"
        if os.path.exists(side_p):
            with open(side_p, "r", encoding="utf-8") as f:
                side = json.load(f)
            if side.get("sha256") and side["sha256"] != sha:
                raise ResumeRejected("sidecar sha mismatch")
        with open(ck, "rb") as f:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        return ptr, obj

    def _verify_and_restore(self, ptr, obj, model, opt, norm, init_sha):
        if obj.get("format") != FORMAT:
            raise ResumeRejected(
                "old/unsupported checkpoint schema %r: batch58 resume REQUIRES %r (a b57 mid "
                "checkpoint's `pos` points at the ALREADY processed batch -> refusing to replay it)"
                % (obj.get("format"), FORMAT))
        if "next_pos" not in obj or "next_batch_ids" not in obj or "last_batch_ids" not in obj:
            raise ResumeRejected("checkpoint lacks next_pos/next_batch_ids/last_batch_ids (v2 schema required)")
        for field, expect in (("config_sha256", config_sha256()),
                              ("datahash", self.cohort.datahash),
                              ("source_sha256", self.cohort.source_sha256),
                              ("target_column", TARGET_COLUMN),
                              ("effective_batch", self.eff),
                              ("seed", SEED)):
            if obj.get(field) != expect:
                raise ResumeRejected("resume mismatch on %s: %r != %r" % (field, obj.get(field), expect))
        if init_sha is not None and obj.get("init_sha") != init_sha:
            raise ResumeRejected("resume init-sha mismatch: %r != %r" % (obj.get("init_sha"), init_sha))
        if bool(obj.get("device_is_cuda")) != bool(str(self.device).startswith("cuda")):
            raise ResumeRejected("resume device-class mismatch: %r vs %r" % (obj.get("device"), self.device))
        model.load_state_dict(obj["model"])
        opt.load_state_dict(obj["optimizer"])
        restore_rng(obj["rng"])
        ns = obj.get("normalizer") or {}
        norm.mean = float(ns.get("mean", 0.0))
        norm.std = float(ns.get("std", 1.0))
        norm.frozen = bool(ns.get("frozen", True))
        ck_epoch = int(obj["epoch"]); ck_next = int(obj["next_pos"])
        if ck_next > self.n:
            raise ResumeRejected("next_pos %d > n_train %d" % (ck_next, self.n))
        rollover = (ck_next == self.n)
        perm = self._perm_of(ck_epoch)
        if rollover:
            # epoch completed; the next unit of work is epoch+1 from position 0
            if list(map(int, obj["next_batch_ids"])) != []:
                raise ResumeRejected("epoch-end checkpoint must carry empty next_batch_ids")
            last_start = self._prev_batch_start(self.n)
            last_ids = perm[last_start:self.n]
            if list(map(int, obj["last_batch_ids"])) != list(map(int, last_ids)):
                raise ResumeRejected("epoch-end last-batch mismatch")
            if obj.get("coverage_hash") and obj["coverage_hash"] != _expected_epoch_coverage(perm):
                raise ResumeRejected("epoch-end coverage hash mismatch (epoch not a full pass)")
            return ck_epoch + 1, 0
        next_ids = perm[ck_next:ck_next + self.eff]
        if ck_next == 0:
            last_ids = []
        else:
            last_start = self._prev_batch_start(ck_next)
            last_ids = perm[last_start:ck_next]
        if list(map(int, obj["next_batch_ids"])) != list(map(int, next_ids)):
            raise ResumeRejected("sampler next-batch mismatch: saved %r != recomputed %r"
                                 % (obj["next_batch_ids"][:5], list(next_ids)[:5]))
        if list(map(int, obj["last_batch_ids"])) != list(map(int, last_ids)):
            raise ResumeRejected("sampler last-batch mismatch (would replay/skip)")
        h = hashlib.sha256(); h.update(b"b58-epoch-coverage|")
        for bi in range(0, ck_next, self.eff):
            h.update(np.asarray(perm[bi:min(bi + self.eff, ck_next)], dtype=np.int64).tobytes())
        if obj.get("coverage_hash") and obj["coverage_hash"] != h.hexdigest():
            raise ResumeRejected("coverage prefix hash mismatch (skip/repeat detected)")
        return ck_epoch, ck_next

    def _snapshot_pre_step(self, model, opt):
        return {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "optimizer": {k: {kk: (vv.clone() if torch.is_tensor(vv) else vv)
                              for kk, vv in v.items()}
                          for k, v in opt.state_dict()["state"].items()},
            "param_groups": [dict(g) for g in opt.state_dict()["param_groups"]],
            "rng": capture_rng(),
        }

    def _restore_pre_step(self, model, opt, snap):
        model.load_state_dict(snap["model"])
        opt.load_state_dict({"state": snap["optimizer"], "param_groups": snap["param_groups"]})
        restore_rng(snap["rng"])
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_batch(self, model, opt, perm, bi):
        idx_list = perm[bi:bi + self.eff]
        if not idx_list:
            return 0.0, 0
        snap = self._snapshot_pre_step(model, opt)
        micro = min(self.micro, len(idx_list))
        attempt = 0
        while True:
            attempt += 1
            opt.zero_grad(set_to_none=True)
            restore_rng(snap["rng"])
            total = 0.0
            try:
                for mb in range(0, len(idx_list), micro):
                    part = idx_list[mb:mb + micro]
                    t = torch.tensor(part, dtype=torch.long)
                    mcodes = self.tr_codes[t]
                    my = self.tr_yz[t]
                    mw = self.tr_w[t]
                    if str(self.device).startswith("cuda"):
                        assert_cuda_all(model, mcodes)   # real-run CUDA contract
                    out = model(mcodes)
                    loss, _parts = losses.total_loss(out, my, mw)
                    (loss * (len(part) / float(self.eff))).backward()
                    total += float(loss.detach()) * len(part)
                break
            except torch.cuda.OutOfMemoryError as e:
                self.oom_retries += 1
                self.oom_events.append({"bi": int(bi), "attempt": attempt, "micro": micro,
                                        "error": str(e)[:200], "ts": now_iso()})
                self._restore_pre_step(model, opt, snap)
                if micro <= 1:
                    raise OOMExhausted("OOM at micro=1 for batch@%d; no CPU neural fallback" % bi)
                micro = max(1, micro // 2)
                if self.micro > micro:
                    self.micro = micro
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        return total, len(idx_list)

    @torch.no_grad()
    def eval_split(self, model, norm, split):
        model.eval()
        idx = self.cohort.split_indices(split)
        preds = []
        for s in range(0, len(idx), 256):
            preds.append(norm.inverse(model(self.cohort.codes[idx[s:s + 256]].to(self.device))).cpu())
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

    def _stop_reason(self):
        if self._should_stop():
            return "signal_or_stopfile"
        if self.stop_after_opt_steps and self.opt_steps >= int(self.stop_after_opt_steps):
            return "stop_after_opt_steps"
        if self.stop_after_seconds and (time.time() - self._t0) >= float(self.stop_after_seconds):
            return "stop_after_seconds"
        return None

    def run(self, purpose="b58-Nabs-throughput-variant"):
        gate, token = self._reserve(purpose)
        self._install_signals()
        self._t0 = time.time()
        try:
            model, opt, norm, init_sha = self._build()
            if self.expected_init_sha and init_sha != self.expected_init_sha:
                gate.fail(token, "init-sha mismatch")
                raise AssertionError("init sha %s != frozen %s" % (init_sha, self.expected_init_sha))
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.opt_steps = 0
            self.global_samples = 0
            self.epoch_samples = 0
            self.pair_ptr = 0
            self.pair_repeats = 0
            self._perm = None
            self._perm_epoch = None
            history = []
            best = {"pair_mae": float("inf"), "epoch": -1, "unset": True}
            start_epoch, start_pos = 0, 0
            resumed = False
            resumed_from = None
            if self.resume_latest:
                ptr, obj = self._load_latest(self.resume_latest)
                start_epoch, start_pos = self._verify_and_restore(ptr, obj, model, opt, norm, init_sha)
                self.opt_steps = int(obj.get("opt_steps", 0))
                self.global_samples = int(obj.get("global_samples", 0))
                self.epoch_samples = int(obj.get("epoch_samples", 0))
                self.pair_ptr = int(obj.get("pair_ptr", 0))
                self.pair_repeats = int(obj.get("pair_repeats", 0))
                history = list(obj.get("history", []))
                best = obj.get("best", best)
                self._last_gen = int(obj.get("generation", 0))
                self.oom_retries = int(obj.get("oom_retries", 0))
                resumed = True
                resumed_from = {"path": ptr.get("path"), "sha256": ptr.get("sha256"),
                                "generation": self._last_gen, "epoch": start_epoch,
                                "next_pos": start_pos, "carried_opt_steps": self.opt_steps,
                                "carried_global_samples": self.global_samples}
            self.last_save = time.time()
            epoch = start_epoch
            cov = hashlib.sha256(); cov.update(b"b58-epoch-coverage|")
            if resumed and start_pos > 0:
                perm0 = self._perm_of(epoch)
                for bi in range(0, start_pos, self.eff):
                    cov.update(np.asarray(perm0[bi:min(bi + self.eff, start_pos)],
                                          dtype=np.int64).tobytes())
            stop_marker = None
            while epoch < self.epoch_cap:
                perm = self._perm_of(epoch)
                pos = start_pos if epoch == start_epoch else 0
                ep_start = time.time()
                last_batch_ids = []
                while pos < self.n:
                    _total, nproc = self._run_batch(model, opt, perm, pos)
                    self.opt_steps += 1
                    self.global_samples += nproc
                    self.epoch_samples += nproc
                    last_batch_ids = perm[pos:pos + nproc]
                    cov.update(np.asarray(last_batch_ids, dtype=np.int64).tobytes())
                    pos += nproc
                    next_ids = perm[pos:pos + self.eff]
                    reason = self._stop_reason()
                    if reason:
                        self._save_generation(model, opt, norm, init_sha, epoch, pos, history, best,
                                              cov.hexdigest(), last_batch_ids, next_ids,
                                              graceful=True, attempt_id=token["exec_id"])
                        stop_marker = {"reason": reason, "epoch": epoch, "next_pos": pos,
                                       "opt_steps": self.opt_steps, "saved": True}
                        gate.complete(token, identity=self.identity() + "|graceful_stop")
                        return self._result(resumed, resumed_from, history, best, init_sha,
                                            epoch, pos, stop_marker, token)
                    if time.time() - self.last_save >= self.snapshot_interval_s:
                        self._save_generation(model, opt, norm, init_sha, epoch, pos, history, best,
                                              cov.hexdigest(), last_batch_ids, next_ids,
                                              attempt_id=token["exec_id"])
                        self.last_save = time.time()
                cov_ok = (cov.hexdigest() == _expected_epoch_coverage(perm))
                epoch_seconds = time.time() - ep_start
                dev_act, dev_pair_mae, dev_sign = self.eval_split(model, norm, "DEV")
                if dev_pair_mae != dev_pair_mae:
                    dev_pair_mae = float("inf")
                if best["unset"] or dev_pair_mae < best["pair_mae"]:
                    best = {"pair_mae": dev_pair_mae, "epoch": epoch, "unset": False,
                            "act_mae": dev_act["mae"]}
                history.append({"epoch": epoch, "epoch_samples": self.epoch_samples,
                                "opt_steps": self.opt_steps, "coverage_ok": cov_ok,
                                "coverage_hash": cov.hexdigest(), "epoch_seconds": epoch_seconds,
                                "dev": dev_act, "dev_pair_mae": dev_pair_mae, "dev_sign": dev_sign})
                self._save_generation(model, opt, norm, init_sha, epoch, self.n, history, best,
                                      cov.hexdigest(), last_batch_ids, [], epoch_end=True,
                                      attempt_id=token["exec_id"], dev=dev_act,
                                      extra={"coverage_ok": cov_ok, "epoch_seconds": epoch_seconds})
                self.last_save = time.time()
                reason = ("stop_after_seconds" if (self.stop_after_seconds and
                          (time.time() - self._t0) >= float(self.stop_after_seconds)) else None)
                if reason:
                    stop_marker = {"reason": reason + "_at_epoch_end", "epoch": epoch,
                                   "next_pos": self.n, "opt_steps": self.opt_steps, "saved": True}
                    break
                epoch += 1
                self.epoch_samples = 0
                cov = hashlib.sha256(); cov.update(b"b58-epoch-coverage|")
                self._perm_epoch = None
            gate.complete(token, identity=self.identity())
            return self._result(resumed, resumed_from, history, best, init_sha, epoch,
                                self.n, stop_marker, token)
        except BaseException as e:
            try:
                gate.fail(token, e)
            except Exception:
                pass
            raise

    def _result(self, resumed, resumed_from, history, best, init_sha, epoch, next_pos,
                stop_marker, token):
        elapsed = (time.time() - self._t0) if self._t0 else None
        return {
            "arm": "Nabs", "variant": "throughput_variant", "run_id": self.run_id,
            "effective_batch": self.eff, "micro_batch": self.micro, "epoch_cap": self.epoch_cap,
            "init_sha": init_sha, "history": history, "best": best,
            "opt_steps": self.opt_steps, "global_samples": self.global_samples,
            "oom_retries": self.oom_retries, "oom_events": self.oom_events,
            "resumed": resumed, "resumed_from": resumed_from,
            "stopped": stop_marker is not None, "stop_marker": stop_marker,
            "checkpoint_latest": os.path.join(self.out_dir, LATEST),
            "last_generation": self._last_gen,
            "epochs_completed": sum(1 for h in history if h.get("coverage_ok")),
            "elapsed_s": elapsed,
            "samples_per_s": (self.global_samples / elapsed) if elapsed else None,
            "epoch": epoch, "next_pos": next_pos,
            "peak_cuda_mb": (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0,
            "attempt_id": token.get("exec_id"),
            "dev_last": history[-1] if history else None,
        }


def main(argv=None):
    import argparse
    from cohort_v2 import load_cohort_v2
    from spec import SOURCE_SHA256
    ap = argparse.ArgumentParser(description="batch58 throughput resource-variant trainer (real resume)")
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--source-sha", default=SOURCE_SHA256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--effective-batch", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--epoch-cap", type=int, default=2)
    ap.add_argument("--run-id", default="Nabs_throughput_variant")
    ap.add_argument("--expected-init-sha", default=None)
    ap.add_argument("--stopfile", default=None)
    ap.add_argument("--snapshot-interval-s", type=float, default=60.0)
    ap.add_argument("--stop-after-opt-steps", type=int, default=None)
    ap.add_argument("--stop-after-seconds", type=float, default=None)
    ap.add_argument("--resume-latest", default=None, help="path to a previous run's latest.json")
    ap.add_argument("--accepted-data-sha", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--i-understand-this-is-a-real-fit", action="store_true", dest="opt_in")
    a = ap.parse_args(argv)
    cohort = load_cohort_v2(a.cohort_dir, source_sha256=a.source_sha, mmap_codes=True)
    tr = B58Trainer(cohort, a.out_dir, a.ledger, device=a.device, effective_batch=a.effective_batch,
                    epoch_cap=a.epoch_cap, micro_batch=a.micro_batch, opt_in=a.opt_in,
                    accepted_data_sha=a.accepted_data_sha, run_id=a.run_id,
                    expected_init_sha=a.expected_init_sha, stopfile=a.stopfile,
                    snapshot_interval_s=a.snapshot_interval_s,
                    stop_after_opt_steps=a.stop_after_opt_steps,
                    stop_after_seconds=a.stop_after_seconds, resume_latest=a.resume_latest)
    purpose = "b58-resume-continuation" if a.resume_latest else "b58-Nabs-throughput-variant"
    res = tr.run(purpose=purpose)
    if a.result_json:
        atomic_write_json(a.result_json, res)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return res


if __name__ == "__main__":
    main()
