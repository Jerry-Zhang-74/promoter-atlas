#!/usr/bin/env python3
"""Poll for completed species downloads and build their promoters (idempotent).

Runs until all SPECIES have both memmap index files, then exits.
"""
import subprocess, time
from pathlib import Path

ROOT = Path(r"E:\promoter_atlas")
MEMDIR = ROOT / "data" / "multispecies" / "memmap"
SPECIES = ["pan_troglodytes", "macaca_mulatta", "callithrix_jacchus",
           "mus_musculus", "rattus_norvegicus", "cavia_porcellus",
           "oryctolagus_cuniculus", "bos_taurus", "sus_scrofa", "ovis_aries",
           "capra_hircus", "equus_caballus", "canis_lupus_familiaris",
           "felis_catus", "pteropus_vampyrus", "loxodonta_africana",
           "monodelphis_domestica", "ornithorhynchus_anatinus"]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
BUILDER = str(ROOT / "runs" / "scripts" / "build_multispecies_promoters.py")

def has_fa(sp):
    d = ROOT / "data" / "genomes" / sp
    return any(d.glob("*.dna.toplevel.fa.gz"))

def built(sp):
    return (MEMDIR / f"{sp}_up2000_index.tsv").exists() and (MEMDIR / f"{sp}_sym400_index.tsv").exists()

done = [sp for sp in SPECIES if built(sp)]
print("already built:", done, flush=True)
while True:
    pending = [sp for sp in SPECIES if not built(sp)]
    if not pending:
        print("ALL SPECIES BUILT", flush=True)
        break
    built_now = False
    for sp in pending:
        if has_fa(sp):
            print("building", sp, flush=True)
            rc = subprocess.run([PY, BUILDER, "--species", sp], capture_output=True, text=True)
            print(sp, "rc", rc.returncode, rc.stdout.strip().split(chr(10))[-1] if rc.stdout else "", flush=True)
            if rc.stderr.strip():
                print(sp, "ERR", rc.stderr.strip()[-200:], flush=True)
            built_now = True
    if not built_now:
        print("waiting for downloads...", flush=True)
    time.sleep(600)
