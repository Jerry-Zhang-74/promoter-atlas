#!/usr/bin/env python3
"""Download v6.6.0 model weights from WEIGHTS_MANIFEST.json."""

import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    manifest_path = repo_root / "artifacts" / "manifests" / "WEIGHTS_MANIFEST.json"
    if not manifest_path.is_file():
        print("ERROR: WEIGHTS_MANIFEST.json not found", file=sys.stderr)
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: invalid manifest: {e}", file=sys.stderr)
        return 1
    if manifest.get("version") != "v6.6.0":
        print("ERROR: only v6.6.0 supported", file=sys.stderr)
        return 1
    target_rel = manifest["checkpoint_file"]
    if target_rel != "models/v6.6.0.pt":
        print("ERROR: unexpected checkpoint destination", file=sys.stderr); return 1
    target = repo_root / target_rel
    # Ensure target is inside repo and no symlink
    try:
        target_resolved = target.resolve()
        repo_resolved = repo_root.resolve()
        if not target_resolved.is_relative_to(repo_resolved):
            print("ERROR: target outside repo", file=sys.stderr)
            return 1
        if target.is_symlink():
            print("ERROR: target is symlink", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"ERROR: path check failed: {e}", file=sys.stderr)
        return 1
    expected_sha = manifest["sha256"]
    expected_size = manifest["bytes"]
    url = manifest["download_url"]
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0 or not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        print("ERROR: invalid expected size/hash", file=sys.stderr); return 1
    if not isinstance(url, str) or not url.startswith("https://github.com/Jerry-Zhang-74/promoter-atlas/releases/download/v6.6.0/"):
        print("ERROR: URL is not the expected HTTPS release asset", file=sys.stderr)
        return 1
    # Check existing
    if target.is_file():
        if target.stat().st_size == expected_size and sha256_file(target) == expected_sha:
            print("Weights already valid, skipping download.")
            return 0
        else:
            print("Existing file invalid, will re-download.")
    # Download to temp file in target directory
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_weights_")
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        # Optional proxy
        proxy = None
        if "--proxy" in sys.argv:
            idx = sys.argv.index("--proxy")
            if idx + 1 < len(sys.argv):
                proxy = sys.argv[idx + 1]
        if proxy:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": proxy}))
        else:
            opener = urllib.request.build_opener()
        with opener.open(url, timeout=120) as response, tmp.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        if total != expected_size:
            print(f"ERROR: size mismatch: expected {expected_size}, got {total}", file=sys.stderr)
            return 1
        if sha256_file(tmp) != expected_sha:
            print("ERROR: SHA mismatch", file=sys.stderr)
            return 1
        os.replace(tmp, target)
        print("Weights downloaded successfully.")
        return 0
    except Exception as e:
        print(f"ERROR: download failed: {e}", file=sys.stderr)
        return 1
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    sys.exit(main())
