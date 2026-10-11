# -*- coding: utf-8 -*-
# batch52 neural arm trainer: Nabs (activity) / Njoint (activity + nested-difference).
# Same fresh architecture, same seed, equal data/optimizer/epoch caps. Checkpoint atomic + resume.
import os, sys, json, time, math, argparse, hashlib, random, subprocess
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SeqCNN
from exec_gate import ExecGate

ROOT = r"E:/promoter_atlas"
DATA = os.path.join(ROOT, "data/breakthrough_20261002/batch52")
PREP = os.path.join(DATA, "prep"); COH = os.path.join(DATA, "cohort")
MODELS = os.path.join(DATA, "models"); os.makedirs(MODELS, exist_ok=True)
LEDGER = os.path.join(ROOT, "runs/breakthrough_20261002/batch52/exec_gate_ledger.jsonl")
SEED = 20261004
LMAX = 600; BATCH = 32; PAIRB = 16; EPOCHS = 6; LR = 1e-3; WD = 1e-4; CLIP = 1.0
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["nabs", "njoint"], required=True)
ap.add_argument("--epochs", type=int, default=EPOCHS)
args = ap.parse_args()
ARM = args.arm; EPOCHS = args.epochs

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.set_num_threads(4)
assert torch.cuda.is_available(), "CUDA required"
dev = torch.device("cuda:0")

# ---- data ----
rows = []  # cid, uid, split, length, activity
with open(os.path.join(PREP, "rows.tsv"), encoding="utf-8") as f:
    f.readline()
    for line in f:
        a = line.rstrip("\n").split("\t")
        rows.append((a[1], a[2], a[3], int(a[4]), float(a[5])))
cid2idx = {r[0]: i for i, r in enumerate(rows)}
N = len(rows)
X = np.load(os.path.join(PREP, "onehot.npy"), mmap_mode="r")
lengths = np.array([r[3] for r in rows], dtype=np.int32)
splits = {}
for i, r in enumerate(rows): splits.setdefault(r[2], []).append(i)
TRAIN = np.array(splits["TRAIN"]); DEV = np.array(splits["DEV"]); TEST = np.array(splits["TEST"])
grp_size = {}
for r in rows: grp_size[r[1]] = grp_size.get(r[1], 0) + 1

def load_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            a = line.rstrip("\n").split("\t")
            d = dict(zip(hdr, a))
            out.append((d["union_id"], d["split"], d["short_coord_id"], int(d["short_len"]),
                        d["long_coord_id"], int(d["long_len"]), float(d["short_log2norm_hek"]),
                        float(d["long_log2norm_hek"])))
    return out
allpairs = load_pairs(os.path.join(COH, "pairs_train_dev.tsv"))
TRAIN_P = [p for p in allpairs if p[1] == "TRAIN"]
DEV_P = [p for p in allpairs if p[1] == "DEV"]
log("rows", N, "TRAIN", len(TRAIN), "DEV", len(DEV), "pairs TRAIN", len(TRAIN_P), "DEV", len(DEV_P))

# ---- normalizers (TRAIN only) ----
gw = np.array([1.0/grp_size[rows[i][1]] for i in TRAIN]); gw = gw/gw.mean()
acts_tr = np.array([rows[i][4] for i in TRAIN])
wmean = float((gw*acts_tr).sum()/gw.sum()); wstd = float(np.sqrt((gw*(acts_tr-wmean)**2).sum()/gw.sum()))
log("target normalizer weighted mean/std", round(wmean, 5), round(wstd, 5))
gc_all = np.zeros(N)
for i in range(N):
    ln = lengths[i]
    sub = np.asarray(X[i, :ln, 1:3]); gc_all[i] = sub.sum()/max(1, ln)
loglen_all = np.log(np.maximum(1, lengths)).astype(np.float64)
ll_mu = float(loglen_all[TRAIN].mean()); ll_sd = float(loglen_all[TRAIN].std() or 1.0)
gc_mu = float(gc_all[TRAIN].mean()); gc_sd = float(gc_all[TRAIN].std() or 1.0)
log("loglen mu/sd", round(ll_mu, 4), round(ll_sd, 4), "gc mu/sd", round(gc_mu, 4), round(gc_sd, 4))

pair_grp = {}
for p in TRAIN_P: pair_grp[p[0]] = pair_grp.get(p[0], 0) + 1
pw = np.array([1.0/pair_grp[p[0]] for p in TRAIN_P]); pw = pw/pw.mean()
log("n TRAIN pair-groups", len(pair_grp), "pair weight mean", round(float(pw.mean()), 4))

def telemetry():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as e:
        return "telemetry_error:%s" % e

log("GPU telemetry:", telemetry())
log("torch", torch.__version__, "cuda", torch.version.cuda, "dev", torch.cuda.get_device_name(0))

model = SeqCNN().to(dev)
for p in model.parameters():
    assert p.device.type == "cuda", "param not on cuda"
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model_sha = hashlib.sha256()
for k in sorted(base_state): model_sha.update(base_state[k].numpy().tobytes())
model_sha = model_sha.hexdigest()
log("initial model params SHA256", model_sha[:16], "nparams", sum(p.numel() for p in model.parameters()))

def build_endpoint_batch(idx):
    B = len(idx); L = LMAX
    xb = np.zeros((B, 4, L), np.float32); mb = np.zeros((B, 1, L), np.float32)
    fb = np.zeros((B, 2), np.float32); yb = np.zeros(B, np.float32); wb = np.zeros(B, np.float32)
    for j, i in enumerate(idx):
        ln = lengths[i]
        xb[j] = X[i, :ln, :].T; mb[j, 0, :ln] = 1.0
        fb[j, 0] = (loglen_all[i]-ll_mu)/ll_sd; fb[j, 1] = (gc_all[i]-gc_mu)/gc_sd
        yb[j] = (rows[i][4]-wmean)/wstd; wb[j] = 1.0/grp_size[rows[i][1]]
    wb = wb/wb.mean()
    t = lambda a, d: torch.as_tensor(a, dtype=d).to(dev)
    return t(xb, torch.float32), t(mb, torch.float32), t(fb, torch.float32), t(yb, torch.float32), t(wb, torch.float32)

def build_pair_batch(pairs):
    si = np.array([cid2idx[p[2]] for p in pairs]); li = np.array([cid2idx[p[4]] for p in pairs])
    xs, ms, fs, ys, _ = build_endpoint_batch(si)
    xl, ml, fl, yl, _ = build_endpoint_batch(li)
    pwv = np.array([1.0/pair_grp[p[0]] for p in pairs]); pwv = pwv/pwv.mean()
    dw = torch.as_tensor((ys-yl).cpu().numpy(), dtype=torch.float32).to(dev)
    return xs, ms, fs, xl, ml, fl, dw, torch.as_tensor(pwv, dtype=torch.float32).to(dev)

def dev_pair_mae(model):
    model.eval(); gp = {}
    with torch.no_grad():
        for s in range(0, len(DEV_P), 64):
            ch = DEV_P[s:s+64]
            si = np.array([cid2idx[p[2]] for p in ch]); li = np.array([cid2idx[p[4]] for p in ch])
            xs, ms, fs, _, _ = build_endpoint_batch(si); xl, ml, fl, _, _ = build_endpoint_batch(li)
            ps = model(xs, ms, fs).cpu().numpy()*wstd
            pl = model(xl, ml, fl).cpu().numpy()*wstd
            dpred = ps-pl
            for j, p in enumerate(ch):
                dtrue = p[6]-p[7]
                gp.setdefault(p[0], []).append(abs(dpred[j]-dtrue))
    model.train()
    per = [np.mean(v) for v in gp.values()]
    return float(np.mean(per)), len(gp)

def save_ckpt(path, epoch, gstep, sampler_pos, best, extra=None):
    d = {"arm": ARM, "epoch": epoch, "global_step": gstep, "sampler_pos": sampler_pos,
         "model": model.state_dict(), "optimizer": opt.state_dict(),
         "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
         "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
         "wmean": wmean, "wstd": wstd, "ll_mu": ll_mu, "ll_sd": ll_sd, "gc_mu": gc_mu, "gc_sd": gc_sd,
         "seed": SEED, "config": {"lr": LR, "wd": WD, "batch": BATCH, "pairb": PAIRB, "epochs": EPOCHS,
         "clip": CLIP, "loss": ARM}, "best_dev_pair_mae": best, "init_model_sha": model_sha}
    if extra: d.update(extra)
    tmp = path + ".tmp"; torch.save(d, tmp); os.replace(tmp, path)

EPOCHS = args.epochs
log("==== TRAIN %s epochs=%d ====" % (ARM, EPOCHS))
best = float("inf"); best_epoch = -1; best_path = os.path.join(MODELS, "ckpt_%s_best.pt" % ARM)
hist = []

def run(gate_token):
    global best, best_epoch
    gstep = 0
    for ep in range(1, EPOCHS+1):
        rng = np.random.default_rng(SEED*1000+ep)
        order = rng.permutation(TRAIN)
        prng = np.random.default_rng(SEED*2000+ep)
        porder = [TRAIN_P[i] for i in prng.permutation(len(TRAIN_P))]
        ppos = 0; last_save = time.time()
        ep_loss = 0.0; nb = 0; pair_repeats = 0
        for s in range(0, len(order), BATCH):
            idx = order[s:s+BATCH]
            xb, mb, fb, yb, wb = build_endpoint_batch(idx)
            pred = model(xb, mb, fb)
            per = F.smooth_l1_loss(pred, yb, beta=1.0, reduction="none")
            loss = (per*wb).sum()/wb.sum()
            if ARM == "njoint":
                pl = porder[ppos:ppos+PAIRB]
                if len(pl) < PAIRB:
                    pl = pl + porder[:(PAIRB-len(pl))]; pair_repeats += 1
                ppos += PAIRB
                if ppos >= len(porder): ppos = 0
                xs, ms, fs, xl, ml, fl, dw, pwv = build_pair_batch(pl)
                dps = model(xs, ms, fs); dpl = model(xl, ml, fl)
                diff = dps - dpl
                ploss = F.smooth_l1_loss(diff, dw, beta=1.0, reduction="none")
                loss = loss + 0.5*(ploss*pwv).sum()/pwv.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step(); gstep += 1; ep_loss += float(loss); nb += 1
            if time.time()-last_save > 300:
                save_ckpt(os.path.join(MODELS, "ckpt_%s_last.pt" % ARM), ep-1, gstep, ppos, best,
                          {"intra_epoch": True})
                last_save = time.time()
        dmae, ng = dev_pair_mae(model)
        hist.append({"epoch": ep, "train_loss": ep_loss/max(1, nb), "dev_group_pair_mae": dmae,
                     "dev_groups": ng, "steps": gstep, "pair_cycler_repeats": pair_repeats})
        log("epoch", ep, "loss", round(ep_loss/max(1, nb), 5), "dev_group_pair_mae", round(dmae, 5),
            "gstep", gstep, "repeats", pair_repeats)
        if dmae < best - 1e-12:
            best = dmae; best_epoch = ep
            save_ckpt(best_path, ep, gstep, ppos, best, {"selected": True})
            log("  new best ->", round(best, 5))
        save_ckpt(os.path.join(MODELS, "ckpt_%s_last.pt" % ARM), ep, gstep, ppos, best)
    return {"best_dev_pair_mae": best, "best_epoch": best_epoch, "history": hist, "steps": gstep}

gate = ExecGate(LEDGER, 4)
st = gate.status(); log("gate status", st)
if st["remaining"] <= 0:
    log("EXEC GATE BUDGET EXHAUSTED"); sys.exit(4)
token = gate.reserve("batch52-%s" % ARM)
log("reserved gate slot", token["exec_id"], "purpose", token["purpose"], "consumes", token["consumes"])
data_sha = hashlib.sha256(("%s|%s|%s" % (N, len(TRAIN_P), model_sha)).encode()).hexdigest()
part1 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
pid = os.getpid()
t_start = time.time()
try:
    result = run(token)
except BaseException as e:
    gate.fail(token, repr(e)[:300]); log("TRAINING FAILED", repr(e)[:300]); raise
elapsed = time.time()-t_start
gate.complete(token, identity=model_sha)
meta = {"arm": ARM, "version": "v6.38.0-full-hek-nested-multitask.rc1", "exec_gate_exec_id": token["exec_id"],
        "status": "complete", "epochs_run": result["history"][-1]["epoch"] if result["history"] else 0,
        "best_epoch": result["best_epoch"], "best_dev_group_pair_mae": result["best_dev_pair_mae"],
        "history": result["history"], "global_steps": result["steps"], "elapsed_s": round(elapsed, 1),
        "start_clock": part1, "end_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": pid,
        "gpu": torch.cuda.get_device_name(0), "telemetry_start": telemetry(),
        "config": {"lr": LR, "wd": WD, "batch": BATCH, "pairb": PAIRB, "epochs": EPOCHS, "clip": CLIP,
                   "seed": SEED, "arch": "stem9-64 + 3x(dil1,2,4) res blocks + masked mean/max + mlp130-64-1"},
        "normalizers": {"wmean": wmean, "wstd": wstd, "ll_mu": ll_mu, "ll_sd": ll_sd, "gc_mu": gc_mu, "gc_sd": gc_sd},
        "init_model_sha": model_sha, "data_sha": data_sha, "n_train_rows": int(len(TRAIN)),
        "n_dev_pairs": len(DEV_P), "n_train_pairs": len(TRAIN_P)}
json.dump(meta, open(os.path.join(MODELS, "%s_meta.json" % ARM), "w"), indent=2)
log("DONE", ARM, "best_dev", round(best, 5), "at epoch", best_epoch, "elapsed", round(elapsed, 1))
