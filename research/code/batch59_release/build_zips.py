# -*- coding: utf-8 -*-
"""batch59 worker: build FLAT competition ZIPs from the reviewed releases.

ZIP root carries the entry directly: predict.py at root + pinned_core/ at root (predict.py
resolves pinned_core from its own directory). No __pycache__/.pyc/temp/out-data/secrets.
Read-only w.r.t. source; ZERO fit/GPU/network.
"""
from __future__ import annotations
import hashlib, json, os, shutil, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch59")
ARMS = {
    "Nabs": (os.path.join(ROOT, "releases", "batch52_first_model", "v6.38.0-nabs-reviewed"),
             "v6.38.0-nabs-reviewed-flat.zip"),
    "Njoint": (os.path.join(ROOT, "releases", "batch52_second_model", "v6.38.0-njoint-reviewed"),
               "v6.38.0-njoint-reviewed-flat.zip"),
}
EXCLUDE_NAMES = {"SHA256SUMS.txt"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def stage_flat(src, flat):
    if os.path.exists(flat):
        shutil.rmtree(flat)
    os.makedirs(flat)
    # predict.py at root; pinned_core at root (so `python predict.py --bundle .` works)
    shutil.copy2(os.path.join(src, "code", "predict.py"), os.path.join(flat, "predict.py"))
    shutil.copytree(os.path.join(src, "code", "pinned_core"), os.path.join(flat, "pinned_core"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("MODEL_MANIFEST.json", "README.md", "USER_GUIDE.md", "USER_GUIDE_zh.md",
                 "MODEL_CARD_zh.md", "LICENSE_NOTES.md", "requirements.txt", "STAGING_STATUS.json"):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(flat, name))
    for d in ("model", "config", "normalizer", "results"):
        s = os.path.join(src, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(flat, d), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def write_sums_flat(flat):
    lines = []
    for dirpath, dirs, files in os.walk(flat):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, flat).replace(os.sep, "/")
            if rel == "SHA256SUMS.txt":
                continue
            lines.append("%s  %s" % (sha256_file(full), rel))
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    open(os.path.join(flat, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    return len(lines)


def make_zip(flat, zippath):
    if os.path.exists(zippath):
        os.remove(zippath)
    members = []
    with zipfile.ZipFile(zippath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirs, files in os.walk(flat):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, flat).replace(os.sep, "/")
                if rel.endswith(".pyc"):
                    continue
                z.write(full, rel)
                members.append(rel)
    return members


def main():
    out = {"kind": "batch59_flat_zip_build", "arms": {}}
    for arm, (src, zname) in ARMS.items():
        flat = os.path.join(REP, "_flat_" + arm)
        stage_flat(src, flat)
        n = write_sums_flat(flat)
        zippath = os.path.join(os.path.dirname(src), zname)  # release area
        members = make_zip(flat, zippath)
        with zipfile.ZipFile(zippath) as z:
            bad = z.testzip()
        out["arms"][arm] = {"zip": os.path.relpath(zippath, ROOT).replace(os.sep, "/"),
                            "zip_sha256": sha256_file(zippath),
                            "members": len(members), "files_hashed": n,
                            "has_root_predict": "predict.py" in members,
                            "has_root_requirements": "requirements.txt" in members,
                            "has_root_readme": "README.md" in members,
                            "no_pycache": not any("__pycache__" in m or m.endswith(".pyc") for m in members),
                            "zip_testzip_ok": bad is None,
                            "members_list": members}
    json.dump(out, open(os.path.join(REP, "zip_build.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({a: {k: v for k, v in d.items() if k != "members_list"} for a, d in out["arms"].items()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
