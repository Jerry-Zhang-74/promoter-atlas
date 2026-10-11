# -*- coding: utf-8 -*-
# batch52: bounded parallel range-download of OFFICIAL Ensembl GRCh37 (= hg19 assembly)
# primary_assembly FASTA. UCSC hg19.2bit was preferred but the UCSC host refused
# connections after a bounded parallel attempt (see reference_provenance).
import os, sys, json, time, hashlib, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/"
FN   = "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
URL  = BASE + FN
OUTDIR = r"E:/promoter_atlas/data/breakthrough_20261002/batch52/reference"
FINAL = os.path.join(OUTDIR, "GRCh37.primary_assembly.fa.gz")
CHUNKDIR = os.path.join(OUTDIR, "chunks_ens")
CHUNK = 8_000_000
NWORK = 12
MAX_TIME = 300
RETRIES = 4

os.makedirs(CHUNKDIR, exist_ok=True)
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

def head():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {k.lower(): v for k, v in r.headers.items()}

def fetch_chunk(i, start, end):
    path = os.path.join(CHUNKDIR, "%05d.part" % i)
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return (i, want, "cached")
    for attempt in range(1, RETRIES+1):
        cmd = ["curl", "-sS", "--fail", "--max-time", str(MAX_TIME),
               "--retry", "2", "--retry-delay", "2",
               "-H", "Range: bytes=%d-%d" % (start, end), "-o", path, URL]
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) == want:
            return (i, want, "ok")
        log("chunk", i, "attempt", attempt, "rc", p.returncode,
            "got", os.path.getsize(path) if os.path.exists(path) else 0, "/", want,
            p.stderr.decode('utf-8','replace')[:100])
        time.sleep(1.5)
    return (i, -1, "fail")

def main():
    h = head(); size = int(h["content-length"])
    log("HEAD size", size, "etag", h.get("etag"), "lm", h.get("last-modified"),
        "ranges", h.get("accept-ranges"))
    nch = (size + CHUNK - 1)//CHUNK
    ranges = [(i, i*CHUNK, min(i*CHUNK+CHUNK-1, size-1)) for i in range(nch)]
    log("chunks", nch, "workers", NWORK)
    failed = []; done = 0
    with ThreadPoolExecutor(max_workers=NWORK) as ex:
        futs = [ex.submit(fetch_chunk, i, s, e) for (i, s, e) in ranges]
        for f in as_completed(futs):
            i, n, st = f.result(); done += 1
            if n < 0: failed.append(i)
            if done % 15 == 0 or done == nch: log("progress", done, "/", nch, "fail", len(failed))
    if failed:
        log("FAILED", failed)
        json.dump({"status":"failed","failed_chunks":failed,"size":size,"url":URL},
                  open(os.path.join(OUTDIR,"download_status.json"),"w"), indent=2)
        sys.exit(2)
    log("assembling+sha256")
    sha = hashlib.sha256(); total = 0
    with open(FINAL+".part","wb") as out:
        for i, s, e in ranges:
            b = open(os.path.join(CHUNKDIR,"%05d.part"%i),"rb").read()
            assert len(b) == (e-s+1), (i, len(b), e-s+1)
            out.write(b); sha.update(b); total += len(b)
    prov = {"provider":"Ensembl (EMBL-EBI)", "url":URL, "content_length":size,
            "etag":h.get("etag"), "last_modified":h.get("last-modified"),
            "accept_ranges":h.get("accept-ranges"), "actual_bytes_on_disk":total,
            "sha256":sha.hexdigest(), "chunk_bytes":CHUNK, "workers":NWORK,
            "range_requests":nch, "head_requests":1, "retries_used":None,
            "assembly":"GRCh37 (== hg19)", "fetched_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ucsc_preferred_attempt":{"url":"https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit",
              "expected_md5":"bcdbfbe9da62f19bee88b74dabef8cd3","expected_bytes":816241703,
              "outcome":"host refused further connections after bounded parallel attempt; partial bytes retained",
              "bytes_fetched_approx":171000000}}
    json.dump(prov, open(os.path.join(OUTDIR,"reference_provenance.json"),"w"), indent=2)
    if total != size:
        log("SIZE MISMATCH"); sys.exit(3)
    os.replace(FINAL+".part", FINAL)
    log("OK size verified", total, "sha256", sha.hexdigest()[:16])
    for i, s, e in ranges:
        try: os.remove(os.path.join(CHUNKDIR,"%05d.part"%i))
        except OSError: pass

if __name__ == "__main__":
    main()
