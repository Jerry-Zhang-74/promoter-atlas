# -*- coding: utf-8 -*-
"""batch60 worker: build CORRECTED flat-layout release revisions v6.38.1-{arm}-final.

BOUNDED repair of the batch59 review findings:
  F1  LICENSE_NOTES falsely asserted optimizer/RNG state is absent (checkpoint DOES contain it).
  F2  nested reviewed SHA256SUMS used code/-prefixed paths that do not resolve from flat root.
  F3  docs carried stale absolute workspace paths and an old version name.
Base = the actual batch59 reviewed flat ZIP bytes. weight/config/core/predict/normalizer/
results/requirements bytes KEPT byte-identical. ZERO fit/solve/backward/GPU/network.
"""
from __future__ import annotations
import hashlib, json, os, shutil, zipfile

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch60")
DOCS = os.path.join(CODE, "docs")

REVISED = ["LICENSE_NOTES.md", "README.md", "USER_GUIDE.md", "USER_GUIDE_zh.md",
           "MODEL_CARD_zh.md", "MODEL_MANIFEST.json", "STAGING_STATUS.json",
           "SHA256SUMS.txt"]

ARMS = {
    "Nabs": {
        "version": "v6.38.1-nabs-final",
        "ckpt": "Nabs_selected.ckpt",
        "sha": "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
        "out_dir": "releases/batch52_first_model/v6.38.1-nabs-final",
        "base_zip": os.path.join(ROOT, "releases", "batch52_first_model",
                                 "v6.38.0-nabs-reviewed-flat.zip"),
        "ARM_DESC": "first model (Nabs)",
        "ARM_ROLE": "first",
    },
    "Njoint": {
        "version": "v6.38.1-njoint-final",
        "ckpt": "Njoint_selected.ckpt",
        "sha": "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee",
        "out_dir": "releases/batch52_second_model/v6.38.1-njoint-final",
        "base_zip": os.path.join(ROOT, "releases", "batch52_second_model",
                                 "v6.38.0-njoint-reviewed-flat.zip"),
        "ARM_DESC": "second model (Njoint)",
        "ARM_ROLE": "second",
    },
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_text(path, text):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def render(template_name, arm):
    meta = ARMS[arm]
    txt = open(os.path.join(DOCS, template_name), encoding="utf-8").read()
    return (txt.replace("{VERSION}", meta["version"])
               .replace("{ARM_DESC}", meta["ARM_DESC"])
               .replace("{ARM_ROLE}", meta["ARM_ROLE"])
               .replace("{ARM}", arm)
               .replace("{CKPT}", meta["ckpt"])
               .replace("{SHA}", meta["sha"]))


def write_sums(root):
    lines = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in sorted(files):
            if fn.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel == "SHA256SUMS.txt":
                continue
            lines.append("%s  %s" % (sha256_file(full), rel))
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    write_text(os.path.join(root, "SHA256SUMS.txt"), "\n".join(lines) + "\n")
    return len(lines)


def make_zip(root, zippath):
    if os.path.exists(zippath):
        os.remove(zippath)
    members = []
    with zipfile.ZipFile(zippath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in sorted(files):
                if fn.endswith(".pyc"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                z.write(full, rel)
                members.append(rel)
    with zipfile.ZipFile(zippath) as z:
        bad = z.testzip()
    return members, bad


def build_arm(arm):
    meta = ARMS[arm]
    rel_dir = meta["out_dir"]
    dst = os.path.join(ROOT, rel_dir.replace("/", os.sep))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    base = {}
    with zipfile.ZipFile(meta["base_zip"]) as z:
        for name in z.namelist():
            base[name] = z.read(name)
    for name, data in base.items():
        p = os.path.join(dst, name.replace("/", os.sep))
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    for doc in ("LICENSE_NOTES.md", "README.md", "USER_GUIDE.md", "USER_GUIDE_zh.md",
                "MODEL_CARD_zh.md"):
        write_text(os.path.join(dst, doc), render(doc, arm))

    man = json.loads(base["MODEL_MANIFEST.json"].decode("utf-8"))
    man["version"] = meta["version"]
    man.setdefault("staging", {})
    man["staging"]["basis"] = ("batch60 v6.38.1 flat-layout revision of the batch59 reviewed "
                               "bundle (documentation + layout-appropriate SHA256SUMS; "
                               "weight/config/core/predict bytes unchanged)")
    man["staging"]["layout"] = "flat"
    man["staging"]["revision"] = "batch60 v6.38.1-release-layout-fix"
    write_text(os.path.join(dst, "MODEL_MANIFEST.json"),
               json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True))

    st = json.loads(base["STAGING_STATUS.json"].decode("utf-8"))
    st["version"] = meta["version"]
    st["out_dir"] = rel_dir
    st["status"] = "FINAL_FLAT_LAYOUT"
    st["reviewed_by"] = "batch60 worker (layout/documentation repair)"
    st["revision"] = "batch60 v6.38.1-release-layout-fix"
    st.setdefault("packaging_fix", {})["layout_fix"] = (
        "flat root-relative SHA256SUMS; LICENSE_NOTES optimizer/RNG prose corrected; "
        "English/Chinese commands use root predict.py + portable python, no stale absolute path")
    write_text(os.path.join(dst, "STAGING_STATUS.json"),
               json.dumps(st, ensure_ascii=False, indent=2, sort_keys=True))

    n_sums = write_sums(dst)

    zippath = os.path.join(os.path.dirname(dst), meta["version"] + "-flat.zip")
    members, bad = make_zip(dst, zippath)

    changed = []
    identical = 0
    for name, data in base.items():
        if name in REVISED:
            continue
        now = os.path.join(dst, name.replace("/", os.sep))
        got = sha256_file(now)
        if got == sha256_bytes(data):
            identical += 1
        else:
            changed.append({"file": name, "base": sha256_bytes(data), "new": got})

    weight_sha = sha256_file(os.path.join(dst, "model", meta["ckpt"]))
    return {
        "arm": arm, "version": meta["version"], "dir": rel_dir,
        "zip": os.path.relpath(zippath, ROOT).replace(os.sep, "/"),
        "zip_sha256": sha256_file(zippath), "zip_testzip_ok": bad is None,
        "zip_members": len(members),
        "has_root_predict": "predict.py" in members,
        "has_root_pinned_core": any(m.startswith("pinned_core/") for m in members),
        "no_pycache": not any("__pycache__" in m or m.endswith(".pyc") for m in members),
        "files_listed_in_sums": n_sums,
        "revised_files": REVISED,
        "base_files_byte_identical_count": identical,
        "base_files_changed_unexpectedly": changed,
        "weight_sha256": weight_sha, "weight_sha_match": weight_sha == meta["sha"],
        "zip_members_list": members,
    }


def main():
    out = {"kind": "batch60_flat_layout_release_build",
           "fixes": ["F1 LICENSE_NOTES optimizer/RNG prose",
                     "F2 root-relative flat SHA256SUMS",
                     "F3 English/Chinese commands root predict.py + portable python"],
           "basis": "batch59 independently CUDA-verified flat ZIP contents (bytes preserved)",
           "arms": {}}
    for arm in ("Nabs", "Njoint"):
        out["arms"][arm] = build_arm(arm)
    out["all_weights_match"] = all(a["weight_sha_match"] for a in out["arms"].values())
    out["all_non_revised_bytes_identical"] = all(
        not a["base_files_changed_unexpectedly"] for a in out["arms"].values())
    os.makedirs(REP, exist_ok=True)
    write_text(os.path.join(REP, "build.json"),
               json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    brief = {a: {k: v for k, v in d.items() if k != "zip_members_list"} for a, d in out["arms"].items()}
    print(json.dumps({"all_weights_match": out["all_weights_match"],
                      "all_non_revised_bytes_identical": out["all_non_revised_bytes_identical"],
                      "arms": brief}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
