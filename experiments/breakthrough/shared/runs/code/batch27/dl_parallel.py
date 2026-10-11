# -*- coding: utf-8 -*-
"""Bounded resumable parallel-range downloader for GEO/GSE supplementary files.
Enforces a walltime cap; writes <out>.partNN; concatenates; verifies size + sha256.
No credentials; public HTTP only.  argv: url out_path nconn walltime_s [expected_size]
"""
import sys, os, time, threading, hashlib, urllib.request, json

def head_size(url, timeout=20):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length")), r.headers.get("Last-Modified")

def fetch_range(url, start, end, path, deadline, out_err):
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
        with urllib.request.urlopen(req, timeout=20) as r, open(path, "wb") as f:
            while True:
                if time.time() > deadline:
                    out_err.append("deadline"); return
                b = r.read(65536)
                if not b: break
                f.write(b)
    except Exception as e:
        out_err.append(repr(e))

def main():
    url, out, nconn, walltime = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    expected = int(sys.argv[5]) if len(sys.argv) > 5 else None
    t0 = time.time()
    if expected is None:
        expected, lm = head_size(url)
    else:
        lm = None
    print(json.dumps({"url": url, "expected_size": expected, "nconn": nconn}))
    chunk = (expected + nconn - 1) // nconn
    deadline = t0 + walltime
    parts = [] ; errs = []
    threads = []
    for i in range(nconn):
        s = i * chunk; e = min(expected, s + chunk) - 1
        if s > e: continue
        p = "%s.part%03d" % (out, i)
        parts.append((i, p, s, e))
        th = threading.Thread(target=fetch_range, args=(url, s, e, p, deadline, errs))
        th.start(); threads.append(th)
    for th in threads: th.join()
    # completeness check
    got = 0; complete = True; sizes = {}
    for i, p, s, e in parts:
        want = e - s + 1
        have = os.path.getsize(p) if os.path.exists(p) else 0
        sizes[i] = {"have": have, "want": want}
        got += have
        if have != want: complete = False
    if complete:
        with open(out, "wb") as fo:
            for i, p, s, e in sorted(parts):
                with open(p, "rb") as fi:
                    while True:
                        b = fi.read(1 << 20)
                        if not b: break
                        fo.write(b)
        tot = os.path.getsize(out)
        h = hashlib.sha256()
        with open(out, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        print(json.dumps({"complete": True, "size": tot, "sha256": h.hexdigest(),
                          "elapsed_s": round(time.time() - t0, 1), "errors": errs}))
    else:
        print(json.dumps({"complete": False, "bytes_fetched": got, "expected": expected,
                          "elapsed_s": round(time.time() - t0, 1), "part_sizes": sizes,
                          "errors": errs}))

if __name__ == "__main__":
    main()
