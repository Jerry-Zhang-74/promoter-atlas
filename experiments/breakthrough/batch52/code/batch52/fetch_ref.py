# -*- coding: utf-8 -*-
# batch52: bounded parallel range-download of OFFICIAL UCSC hg19 2bit.
# Records provenance; verifies size + official md5; no partial file accepted as complete.
import os, sys, json, time, hashlib, threading, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit"
MD5_EXPECT = "bcdbfbe9da62f19bee88b74dabef8cd3"
OUTDIR = r"E:/promoter_atlas/data/breakthrough_20261002/batch52/reference"
FINAL = os.path.join(OUTDIR, "hg19.2bit")
CHUNKDIR = os.path.join(OUTDIR, "chunks")
CHUNK = 4_000_000
NWORK = 12
MAX_TIME = 300          # per-request seconds
RETRIES = 3

os.makedirs(CHUNKDIR, exist_ok=True)
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)

def head():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {k.lower(): v for k, v in r.headers.items()}

def fetch_chunk(i, start, end, size):
    # return (i, bytes_written) ; writes CHUNKDIR/i.part
    path = os.path.join(CHUNKDIR, "%05d.part" % i)
    want = end - start + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return (i, want, "cached")
    for attempt in range(1, RETRIES+1):
        try:
            cmd = ["curl", "-sS", "--fail", "--max-time", str(MAX_TIME),
                   "--speed-limit", "1000", "--speed-time", "60",
                   "-H", "Range: bytes=%d-%d" % (start, end),
                   "-o", path, URL]
            p = subprocess.run(cmd, capture_output=True)
            if p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) == want:
                return (i, want, "ok")
            log("chunk", i, "attempt", attempt, "rc", p.returncode,
                "got", os.path.getsize(path) if os.path.exists(path) else 0, "/", want,
                p.stderr.decode('utf-8', 'replace')[:120])
        except Exception as e:
            log("chunk", i, "attempt", attempt, "exc", str(e)[:120])
        time.sleep(1.0)
    return (i, -1, "fail")

def main():
    h = head()
    size = int(h["content-length"])
    etag = h.get("etag"); lm = h.get("last-modified")
    log("HEAD", "size", size, "etag", etag, "last-modified", lm,
        "accept-ranges", h.get("accept-ranges"))
    nchunks = (size + CHUNK - 1)//CHUNK
    ranges = []
    for i in range(nchunks):
        s = i*CHUNK; e = min(s+CHUNK-1, size-1)
        ranges.append((i, s, e))
    log("chunks", nchunks, "workers", NWORK)
    done = 0; failed = []
    with ThreadPoolExecutor(max_workers=NWORK) as ex:
        futs = [ex.submit(fetch_chunk, i, s, e, size) for (i, s, e) in ranges]
        for f in as_completed(futs):
            i, n, st = f.result()
            done += 1
            if n < 0:
                failed.append(i)
            if done % 20 == 0 or done == nchunks:
                log("progress", done, "/", nchunks, "failed", len(failed))
    if failed:
        log("FAILED chunks", failed)
        json.dump({"status": "failed", "failed_chunks": failed, "size": size,
                   "url": URL, "etag": etag, "last_modified": lm},
                  open(os.path.join(OUTDIR, "download_status.json"), "w"), indent=2)
        sys.exit(2)
    # verify each chunk identity/length then concatenate contiguously
    log("assembling")
    md5 = hashlib.md5(); total = 0
    with open(FINAL + ".part", "wb") as out:
        for i, s, e in ranges:
            p = os.path.join(CHUNKDIR, "%05d.part" % i)
            b = open(p, "rb").read()
            assert len(b) == (e-s+1), (i, len(b), e-s+1)
            out.write(b); md5.update(b); total += len(b)
    got_md5 = md5.hexdigest()
    log("assembled bytes", total, "md5", got_md5, "expected", MD5_EXPECT)
    prov = {"url": URL, "content_length": size, "etag": etag, "last_modified": lm,
            "accept_ranges": h.get("accept-ranges"),
            "requests": {"head": 1, "range_chunks": nchunks, "retries": 0},
            "chunk_bytes": CHUNK, "workers": NWORK, "per_request_max_s": MAX_TIME,
            "actual_bytes_on_disk": total, "md5": got_md5, "md5_expected": MD5_EXPECT,
            "md5_ok": got_md5 == MD5_EXPECT,
            "sources": {"md5sum_txt": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/md5sum.txt"},
            "assembly": "hg19", "provider": "UCSC", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(prov, open(os.path.join(OUTDIR, "reference_provenance.json"), "w"), indent=2)
    if got_md5 != MD5_EXPECT:
        log("MD5 MISMATCH - keeping .part, not accepting as complete")
        sys.exit(3)
    os.replace(FINAL + ".part", FINAL)
    log("OK md5 verified; final at", FINAL)
    # cleanup chunks dir to bound disk
    for i, s, e in ranges:
        try: os.remove(os.path.join(CHUNKDIR, "%05d.part" % i))
        except OSError: pass

if __name__ == "__main__":
    main()
