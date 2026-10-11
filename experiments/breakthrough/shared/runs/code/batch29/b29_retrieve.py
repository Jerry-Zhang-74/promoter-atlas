# -*- coding: utf-8 -*-
"""BATCH29 sequence retrieval: hg19/GRCh37 reference intervals for the frozen selected cohort.
FORWARD genomic sequence, then ASSAY-ORIENTED = forward for '+', reverse-complement applied
EXACTLY ONCE for '-' (author-verified convention, see source_resolution.json).
Bounded: Ensembl GRCh37 REST, <=3 tries/request, 3 threads.  NO model, NO fit."""
import json, time, os, hashlib, urllib.request, threading, queue

SEL = "data/breakthrough_20261002/batch29/selected_rows.json"
OUTJ = "reports/breakthrough_20261002/batch29/retrieval.json"
FAF = "data/breakthrough_20261002/batch29/sure29_forward.fa"
FAA = "data/breakthrough_20261002/batch29/sure29_assay_oriented.fa"
CACHE = "data/breakthrough_20261002/batch27/sure_seq_sample.fa"
URL = "https://grch37.rest.ensembl.org/sequence/region/human"
CHUNK = 50; NTHREAD = 3; MAXTRY = 3

rows = json.load(open(SEL, encoding="utf-8"))["rows"]
sel = [r["key"] for r in rows]

# reuse cached forward FASTA on EXACT interval match (label-free cache)
# cache header format: >chr21_40798184-40798740_+  (k.replace(':','_'))
cache = {}
if os.path.exists(CACHE):
    k = None
    for line in open(CACHE, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            p = line[1:].split("_")
            k = "%s:%s:%s" % (p[0], p[1], p[2]) if len(p) == 3 else None
        elif k is not None:
            cache[k] = line

def parse(k):
    c, rng, sd = k.split(":"); st, en = rng.split("-"); return c, int(st), int(en), sd
def region(k):
    c, st, en, sd = parse(k); return "%s:%d..%d:1" % (c, st + 1, en)

res = {}; failures = []; lock = threading.Lock(); t0 = time.time(); nbytes = [0]
todo = [k for k in sel if k not in cache]
pos = {k: i for i, k in enumerate(todo)}
print("[retr] cached=%d to_fetch=%d" % (len(sel) - len(todo), len(todo)), flush=True)
jobs = [(i, k) for i, k in enumerate(todo)]
batches = [jobs[i:i + CHUNK] for i in range(0, len(jobs), CHUNK)]
q = queue.Queue()
for b in batches: q.put(b)

def worker():
    while True:
        try: b = q.get_nowait()
        except queue.Empty: return
        regs = [region(k) for _, k in b]
        body = json.dumps({"regions": regs}).encode()
        for attempt in range(MAXTRY):
            try:
                rq = urllib.request.Request(URL, data=body,
                        headers={"Content-Type": "application/json", "Accept": "application/json"})
                raw = urllib.request.urlopen(rq, timeout=90).read()
                ents = json.loads(raw.decode())
                if len(ents) != len(b): raise ValueError("resp %d != req %d" % (len(ents), len(b)))
                with lock:
                    nbytes[0] += len(raw)
                    for (gi, k), ent in zip(b, ents): res[gi] = ent.get("seq", "")
                break
            except Exception as e:
                if attempt == MAXTRY - 1:
                    with lock: failures.append({"key": b[0][1], "err": repr(e)[:200]})
                else: time.sleep(2 + attempt * 2)
        q.task_done()

ths = [threading.Thread(target=worker, daemon=True) for _ in range(NTHREAD)]
for t in ths: t.start()
while any(t.is_alive() for t in ths):
    time.sleep(10)
    print("[retr] done=%d/%d %.0fs bytes=%d" % (len(res), len(todo), time.time() - t0, nbytes[0]), flush=True)
for t in ths: t.join()

RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
fwd = {}; assay = {}; n_len_ok = 0; n_N = 0; n_bad = 0; n_missing = 0; bad = []
for r in rows:
    k = r["key"]; c, st, en, sd = parse(k)
    s = cache.get(k)
    if s is None:
        gi = pos.get(k)
        s = res.get(gi) if gi is not None else None
    if not s: n_missing += 1; continue
    s = s.upper()
    if len(s) != en - st:
        n_bad += 1; bad.append({"key": k, "got": len(s), "want": en - st}); continue
    n_len_ok += 1
    if any(ch not in "ACGT" for ch in s): n_N += 1; continue
    fwd[k] = s
    assay[k] = (s.translate(RC)[::-1] if sd == "-" else s)

with open(FAF, "w", encoding="utf-8", newline="\n") as f:
    for k in sel:
        if k in fwd: f.write(">%s\n%s\n" % (k.replace(":", "_"), fwd[k]))
with open(FAA, "w", encoding="utf-8", newline="\n") as f:
    for k in sel:
        if k in assay: f.write(">%s\n%s\n" % (k.replace(":", "_"), assay[k]))
sha_f = hashlib.sha256(open(FAF, "rb").read()).hexdigest()
sha_a = hashlib.sha256(open(FAA, "rb").read()).hexdigest()
out = {"batch": "batch29",
 "reference": "Ensembl GRCh37 REST /sequence/region/human (hg19), forward strand (:1)",
 "coordinate_convention": "UCSC hg19 0-based half-open [start,end) == Ensembl GRCh37 1-based inclusive start+1..end",
 "orientation_rule": "assay-oriented = forward for '+'; reverse-complement of forward applied EXACTLY ONCE for '-' (author-verified)",
 "n_requested": len(sel), "n_from_cache": len(sel) - len(todo), "n_length_ok": n_len_ok,
 "n_missing": n_missing, "n_length_mismatch": n_bad, "n_containing_nonACGT": n_N,
 "n_usable_forward": len(fwd), "n_usable_assay_oriented": len(assay),
 "n_http_batches": len(batches), "n_failed_batches": len(failures), "failures": failures[:8],
 "response_bytes_total": nbytes[0], "elapsed_s": round(time.time() - t0, 1),
 "forward_fasta": FAF, "forward_sha256": sha_f, "assay_fasta": FAA, "assay_sha256": sha_a,
 "length_examples_bad": bad[:5]}
json.dump(out, open(OUTJ, "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))
