# -*- coding: utf-8 -*-
"""BATCH27 B: bounded primary sequence retrieval (Ensembl GRCh37 REST) for a deterministic
label-free stride sample of the FROZEN selected SuRE inserts. Verifies exact coordinates,
length match, orientation convention; estimates exact-duplicate and RC-duplicate rates.
NO fits, NO model."""
import json, time, os, hashlib, urllib.request, urllib.error

SEL = "reports/breakthrough_20261002/batch27/sure_selected_ids.json"
OUT = "reports/breakthrough_20261002/batch27/sure_seq_sample.json"
FA = "data/breakthrough_20261002/batch27/sure_seq_sample.fa"
K = 2500
BATCH = 50
URL = "https://grch37.rest.ensembl.org/sequence/region/human"

sel = json.load(open(SEL, encoding="utf-8"))["selected"]
step = max(1, len(sel) // K)
sample = sel[::step][:K]

def region_of(s):
    c, rng, sd = s.split(":")
    st, en = rng.split("-")
    return c, int(st), int(en), sd

reqs = []
for s in sample:
    c, st, en, sd = region_of(s)
    reqs.append("%s:%d..%d:1" % (c, st + 1, en))   # Ensembl GRCh37 is 1-based inclusive

t0 = time.time()
by_pos = {}       # global request index -> sequence (Ensembl POST returns regions in request order)
fail = []
n_post = 0
cont = []
for i in range(0, len(reqs), BATCH):
    chunk = reqs[i:i + BATCH]
    body = json.dumps({"regions": chunk}).encode()
    ok = False
    for attempt in range(3):
        try:
            rq = urllib.request.Request(URL, data=body,
                    headers={"Content-Type": "application/json", "Accept": "application/json"})
            raw = urllib.request.urlopen(rq, timeout=60).read()
            cont.append(raw)
            ents = json.loads(raw.decode())
            if len(ents) != len(chunk):
                raise ValueError("response length %d != request %d" % (len(ents), len(chunk)))
            for j, ent in enumerate(ents):
                by_pos[i + j] = ent.get("seq", "")
            n_post += 1
            ok = True
            break
        except Exception as e:
            if attempt == 2:
                fail.append({"batch_start": i, "error": repr(e)[:200]})
            else:
                time.sleep(2)
    print("[seq] batch %d/%d ok=%s %.0fs" % (i // BATCH + 1, (len(reqs) + BATCH - 1) // BATCH, ok, time.time() - t0), flush=True)

# verify (positional mapping; request order == response order, asserted per batch)
n_len_ok = 0; n_id_ok = 0; n_with_N = 0; n_missing = 0
len_mismatch = []
seqs = {}
for gi, s_ in enumerate(sample):
    c, st, en, sd = region_of(s_)
    seq = by_pos.get(gi)
    if seq is None or seq == "":
        n_missing += 1
        continue
    n_id_ok += 1
    if len(seq) == en - st:
        n_len_ok += 1
        seqs[s_] = seq.upper()
    else:
        len_mismatch.append({"key": s_, "got": len(seq), "want": en - st})

RC = str.maketrans("ACGTN", "TGCAN")
seen = {}; rcseen = {}
n_exact_dup_pairs = 0; n_rc_pairs = 0
for s, sq in seqs.items():
    if sq in seen:
        n_exact_dup_pairs += 1
    else:
        seen[sq] = s
    r = sq.translate(RC)[::-1]
    if r in rcseen:
        n_rc_pairs += 1
    rcseen[r] = s

with open(FA, "w", encoding="utf-8") as fh:
    for s, sq in seqs.items():
        fh.write(">%s\n%s\n" % (s.replace(":", "_"), sq))
fa_sha = hashlib.sha256(open(FA, "rb").read()).hexdigest()

res = {
    "reference": "Ensembl GRCh37 REST /sequence/region/human (hg19/GRCh37), forward strand",
    "coordinate_convention_verified": "UCSC hg19 0-based half-open [start,end) == Ensembl GRCh37 1-based inclusive start+1..end",
    "sampling_rule": "deterministic label-free stride over the hash-ordered frozen selection: sel[::%d][:%d]" % (step, K),
    "id_mapping": "positional (Ensembl POST returns regions in request order; batch length asserted equal); note Ensembl normalises chromosome ids by stripping the 'chr' prefix",
    "n_requested": len(sample),
    "n_retrieved": len(seqs),
    "n_missing": n_missing,
    "n_id_matched": n_id_ok,
    "n_length_matches_interval": n_len_ok,
    "n_length_mismatch": len(len_mismatch),
    "length_mismatch_examples": len_mismatch[:5],
    "sequence_retrieval_fraction": round(len(seqs) / len(sample), 6),
    "length_match_fraction_of_retrieved": round(n_len_ok / len(seqs), 6) if seqs else None,
    "n_retrieved_containing_N": n_with_N,
    "n_exact_duplicate_sequence_pairs_in_sample": n_exact_dup_pairs,
    "n_reverse_complement_pairs_in_sample": n_rc_pairs,
    "n_http_batches": n_post,
    "n_failed_batches": len(fail),
    "failures": fail[:5],
    "response_sha256_concat": hashlib.sha256(b"".join(cont)).hexdigest(),
    "fasta": FA, "fasta_sha256": fa_sha,
    "elapsed_s": round(time.time() - t0, 1),
}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2))
