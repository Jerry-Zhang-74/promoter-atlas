# -*- coding: utf-8 -*-
"""batch60 reviewer/worker: independent read-only verification of BOTH corrected archives.

For each arm: extract the FINAL -flat.zip to a FRESH temp dir and, in TWO independent read-only
passes, (a) confirm the zip CRC (testzip), (b) verify EVERY literal SHA256SUMS path/hash
resolves from the extracted root (no code/ stripping), (c) confirm completeness - every
extracted file except SHA256SUMS.txt is listed exactly once, (d) confirm the weights match the
protocol-fixed SHAs, (e) prove byte-identity of weight/config/pinned_core/predict/normalizer/
results/requirements vs the batch59 base ZIP, and (f) confirm the release DIR equals the ZIP.
ZERO fit/solve/backward/GPU/network/prediction. CPU only.
"""
from __future__ import annotations
import hashlib, json, os, tempfile, zipfile, shutil

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CODE, "..", ".."))
REP = os.path.join(ROOT, "reports", "breakthrough_20261002", "batch60")

ARMS = {
    "Nabs": {
        "sha": "17a6f622c3d684743d75263af958cb71dad6f13f99e57f6fc69d5336e74c130f",
        "ckpt": "Nabs_selected.ckpt",
        "dir": "releases/batch52_first_model/v6.38.1-nabs-final",
        "zip": "releases/batch52_first_model/v6.38.1-nabs-final-flat.zip",
        "base": "releases/batch52_first_model/v6.38.0-nabs-reviewed-flat.zip",
    },
    "Njoint": {
        "sha": "c2698adde33155e28eace574e05dd42d4eef737002e0c68084053ea9dccb0cee",
        "ckpt": "Njoint_selected.ckpt",
        "dir": "releases/batch52_second_model/v6.38.1-njoint-final",
        "zip": "releases/batch52_second_model/v6.38.1-njoint-final-flat.zip",
        "base": "releases/batch52_second_model/v6.38.0-njoint-reviewed-flat.zip",
    },
}

REVISED = ["LICENSE_NOTES.md", "README.md", "USER_GUIDE.md", "USER_GUIDE_zh.md",
           "MODEL_CARD_zh.md", "MODEL_MANIFEST.json", "STAGING_STATUS.json", "SHA256SUMS.txt"]
FORBIDDEN = ("__pycache__", ".pyc", ".key", ".pem", "api_key", "email_key", "credential",
             ".fa", ".fasta", ".vcf")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def extract(zip_abs, dest):
    with zipfile.ZipFile(zip_abs) as z:
        bad = z.testzip()              # CRC check
        names = z.namelist()
        z.extractall(dest)
    return bad, names


def verify_extracted(ext, sums_name="SHA256SUMS.txt"):
    sums_path = os.path.join(ext, sums_name)
    with open(sums_path, encoding="utf-8") as f:
        entries = [ln.rstrip("\n") for ln in f if ln.strip()]
    listed = [ln.split("  ", 1)[1] for ln in entries]
    # two independent read-only passes
    passes = []
    per_pass_mismatch = []
    for _ in range(2):
        ok = 0
        mism = []
        for ln in entries:
            want, rel = ln.split("  ", 1)
            full = os.path.join(ext, rel.replace("/", os.sep))
            got = sha256_file(full) if os.path.exists(full) else None
            if got == want:
                ok += 1
            else:
                mism.append({"file": rel, "expected": want, "observed": got})
        passes.append(ok)
        per_pass_mismatch = mism
    # completeness: extracted files (minus sums) == listed, exactly once
    on_disk = []
    for dirpath, dirs, files in os.walk(ext):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            rel = os.path.relpath(os.path.join(dirpath, fn), ext).replace(os.sep, "/")
            on_disk.append(rel)
    on_disk_set = set(on_disk) - {sums_name}
    listed_set = set(listed)
    return {
        "entries": len(entries),
        "pass1_matched": passes[0], "pass2_matched": passes[1],
        "consistent": passes[0] == passes[1] == len(entries),
        "mismatches": per_pass_mismatch,
        "duplicate_entries": sorted(x for x in set(listed) if listed.count(x) > 1),
        "files_on_disk": len(on_disk_set),
        "listed_paths_literal_resolve": listed_set == on_disk_set,
        "unlisted_on_disk": sorted(on_disk_set - listed_set),
        "listed_but_absent": sorted(listed_set - on_disk_set),
    }


def check_arm(arm):
    meta = ARMS[arm]
    zip_abs = os.path.join(ROOT, meta["zip"].replace("/", os.sep))
    dir_abs = os.path.join(ROOT, meta["dir"].replace("/", os.sep))
    base_abs = os.path.join(ROOT, meta["base"].replace("/", os.sep))
    tmp = tempfile.mkdtemp(prefix="b60_%s_" % arm)
    try:
        bad, names = extract(zip_abs, tmp)
        res = {"zip_sha256": sha256_file(zip_abs), "zip_testzip_crc_ok": bad is None,
               "zip_members": len(names)}
        res["sums"] = verify_extracted(tmp)
        # forbidden members
        res["forbidden_members"] = [n for n in names
                                    if any(t in n for t in FORBIDDEN)]
        res["no_pycache_keys_rawdata"] = not res["forbidden_members"]
        res["has_root_predict"] = "predict.py" in names
        res["has_root_pinned_core"] = any(n.startswith("pinned_core/") for n in names)
        # weights + identity vs base zip
        res["weight_sha256"] = sha256_file(os.path.join(tmp, "model", meta["ckpt"]))
        res["weight_sha_match"] = res["weight_sha256"] == meta["sha"]
        with zipfile.ZipFile(base_abs) as zb:
            base = {n: zb.read(n) for n in zb.namelist()}
        changed, identical = [], 0
        for n, data in base.items():
            if n in REVISED:
                continue
            got = sha256_file(os.path.join(tmp, n.replace("/", os.sep)))
            if got == hashlib.sha256(data).hexdigest():
                identical += 1
            else:
                changed.append({"file": n, "base": hashlib.sha256(data).hexdigest(), "new": got})
        res["base_files_byte_identical_count"] = identical
        res["base_files_changed_unexpectedly"] = changed
        res["shared_scientific_bytes_identical"] = not changed
        # release DIR equals ZIP contents
        dir_files, diffs = [], []
        for dirpath, dirs, files in os.walk(dir_abs):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in files:
                rel = os.path.relpath(os.path.join(dirpath, fn), dir_abs).replace(os.sep, "/")
                dir_files.append(rel)
        for rel in sorted(set(dir_files) | set(names)):
            fdir = os.path.join(dir_abs, rel.replace("/", os.sep))
            in_zip = rel in names
            if not os.path.exists(fdir):
                diffs.append({"only_in_zip": rel})
            elif not in_zip:
                diffs.append({"only_in_dir": rel})
            else:
                with zipfile.ZipFile(zip_abs) as z:
                    zb = z.read(rel)
                if sha256_file(fdir) != hashlib.sha256(zb).hexdigest():
                    diffs.append({"hash_differs": rel})
        res["dir_equals_zip"] = not diffs
        res["dir_vs_zip_diffs"] = diffs
        res["dir_files"] = len(dir_files)
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    out = {"kind": "batch60_release_verification", "readonly": True,
           "zero_fit_solve_backward_training": True, "zero_gpu": True, "zero_network": True,
           "zero_new_prediction": True,
           "sums_policy": "SHA256SUMS.txt lists every extracted file exactly once except itself "
                          "(flat, root-relative; no code/ stripping)",
           "arms": {}}
    for arm in ("Nabs", "Njoint"):
        out["arms"][arm] = check_arm(arm)
    out["all_sums_pass"] = all(a["sums"]["consistent"] for a in out["arms"].values())
    out["all_paths_resolve_literally"] = all(a["sums"]["listed_paths_literal_resolve"]
                                             for a in out["arms"].values())
    out["all_covered_exactly_once"] = all(
        a["sums"]["listed_paths_literal_resolve"] and not a["sums"]["duplicate_entries"]
        for a in out["arms"].values())
    out["all_weights_match"] = all(a["weight_sha_match"] for a in out["arms"].values())
    out["all_shared_scientific_bytes_identical"] = all(
        a["shared_scientific_bytes_identical"] for a in out["arms"].values())
    out["all_zip_crc_ok"] = all(a["zip_testzip_crc_ok"] for a in out["arms"].values())
    out["all_dirs_equal_zips"] = all(a["dir_equals_zip"] for a in out["arms"].values())
    out["all_no_pycache_keys_rawdata"] = all(a["no_pycache_keys_rawdata"] for a in out["arms"].values())
    out["PASS"] = all([out["all_sums_pass"], out["all_paths_resolve_literally"],
                       out["all_covered_exactly_once"], out["all_weights_match"],
                       out["all_shared_scientific_bytes_identical"], out["all_zip_crc_ok"],
                       out["all_dirs_equal_zips"], out["all_no_pycache_keys_rawdata"]])
    os.makedirs(REP, exist_ok=True)
    with open(os.path.join(REP, "verification.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k.startswith("all") or k == "PASS"},
                     ensure_ascii=False, indent=2))
    for arm in ("Nabs", "Njoint"):
        s = out["arms"][arm]["sums"]
        print(arm, "entries", s["entries"], "ok", s["pass1_matched"], s["pass2_matched"],
              "literal_resolve", s["listed_paths_literal_resolve"],
              "dup", s["duplicate_entries"])
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
