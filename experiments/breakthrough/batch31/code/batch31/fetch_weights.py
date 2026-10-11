#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Robust resumable fetch of official PARM HEK293 weights via jsDelivr CDN mirror,
verified against the GitHub API git-blob SHA1s at a pinned commit. External code."""
import json, hashlib, os, sys, time, urllib.request

EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "data/breakthrough_20261002/batch31/_external")
SHA = "260a27082942060364c1c4e2393018a0fa128844"
OUTDIR = os.path.join(EXT, "src", "HEK293")
os.makedirs(OUTDIR, exist_ok=True)
tree = json.load(open(os.path.join(EXT, "parm_tree.json"), encoding="utf-8"))
want = {t["path"]: (t["sha"], t["size"]) for t in tree["tree"]
        if t["path"].startswith("pre_trained_models/HEK293/") and t["path"].endswith(".parm")}

def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

def fetch(path):
    rel = path.split("/")[-1]
    dest = os.path.join(OUTDIR, rel)
    exp_sha, exp_size = want[path]
    url = f"https://cdn.jsdelivr.net/gh/vansteensellab/PARM@{SHA}/{path}"
    for att in range(60):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have == exp_size:
            data = open(dest, "rb").read()
            if blob_sha1(data) == exp_sha:
                print(f"OK {rel} size={have} sha1_ok", flush=True); return True
            else:
                print(f"BAD {rel} sha mismatch, refetch", flush=True); os.remove(dest); have = 0
        req = urllib.request.Request(url)
        if have: req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb" and have: have = 0
                with open(dest, mode) as f:
                    while True:
                        b = r.read(65536)
                        if not b: break
                        f.write(b)
        except Exception as e:
            print(f"att{att} {rel} err {e}", flush=True)
            time.sleep(2)
    print(f"FAIL {rel}", flush=True); return False

if __name__ == "__main__":
    ok = True
    for p in sorted(want):
        ok = fetch(p) and ok
    print("ALL_OK" if ok else "INCOMPLETE", flush=True)
