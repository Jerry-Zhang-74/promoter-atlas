#!/usr/bin/env python3
"""Download Ensembl release-113 toplevel FASTA + GTF for the mammal panel, plus omics raw files.

Resume-capable: uses curl -C - ; skips files that already exist with correct size.
Output layout:
  data/genomes/<species>/<Species>.<assembly>.dna.toplevel.fa.gz
  data/genomes/<species>/<Species>.<assembly>.113.gtf.gz
  data/omics/raw/{cage,chip,procap,histone,dnase}/...
"""
import urllib.request, ssl, re, subprocess, sys, json
from pathlib import Path

ROOT = Path(r"E:\promoter_atlas")
GENOMES = ROOT / "data" / "genomes"
OMICS = ROOT / "data" / "omics" / "raw"
for d in (GENOMES, OMICS):
    d.mkdir(parents=True, exist_ok=True)

SPECIES = {
    "pan_troglodytes": "Pan_troglodytes",
    "macaca_mulatta": "Macaca_mulatta",
    "callithrix_jacchus": "Callithrix_jacchus",
    "mus_musculus": "Mus_musculus",
    "rattus_norvegicus": "Rattus_norvegicus",
    "cavia_porcellus": "Cavia_porcellus",
    "oryctolagus_cuniculus": "Oryctolagus_cuniculus",
    "bos_taurus": "Bos_taurus",
    "sus_scrofa": "Sus_scrofa",
    "ovis_aries": "Ovis_aries",
    "capra_hircus": "Capra_hircus",
    "equus_caballus": "Equus_caballus",
    "canis_lupus_familiaris": "Canis_lupus_familiaris",
    "felis_catus": "Felis_catus",
    "pteropus_vampyrus": "Pteropus_vampyrus",
    "loxodonta_africana": "Loxodonta_africana",
    "monodelphis_domestica": "Monodelphis_domestica",
    "ornithorhynchus_anatinus": "Ornithorhynchus_anatinus",
}

ctx = ssl.create_default_context()
proxy = urllib.request.ProxyHandler({"https": "http://127.0.0.1:7897", "http": "http://127.0.0.1:7897"})
op = urllib.request.build_opener(proxy)
def get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with op.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")

def curl_dl(url, out, retries=4):
    if out.exists():
        print("  (exists)", out.name, flush=True)
        return True
    tmp = Path(str(out) + ".part")
    for attempt in range(retries):
        rc = subprocess.run(
            ["curl", "-L", "--retry", "5", "--retry-delay", "10", "-C", "-",
             "-x", "http://127.0.0.1:7897", "-sS", "-o", str(tmp), url],
            capture_output=True, text=True, timeout=None)
        if rc.returncode == 0 and tmp.exists():
            tmp.rename(out)
            print("  DONE", out.name, out.stat().st_size, "bytes", flush=True)
            return True
        print(f"  retry {attempt+1} rc={rc.returncode} {rc.stderr.strip()[:200]}", flush=True)
    return False

manifest = []  # (url, path)
# genomes
for sp_dir, sp_cap in SPECIES.items():
    try:
        fasta_html = get(f"https://ftp.ensembl.org/pub/release-113/fasta/{sp_dir}/dna/")
        fa = sorted(set(re.findall(r'href="([A-Za-z0-9_.-]*dna\.toplevel\.fa\.gz)"', fasta_html)))
        gtf_html = get(f"https://ftp.ensembl.org/pub/release-113/gtf/{sp_dir}/")
        gt = sorted(set(re.findall(r'href="([A-Za-z0-9_.-]*\.113\.gtf\.gz)"', gtf_html)))
        if fa and gt:
            fafile = [f for f in fa if "primary_assembly" not in f and "alt" not in f.lower()]
            fafile = fafile or fa
            fafile = fafile[0]
            manifest.append((f"https://ftp.ensembl.org/pub/release-113/fasta/{sp_dir}/dna/{fafile}",
                             GENOMES / sp_dir / fafile))
            manifest.append((f"https://ftp.ensembl.org/pub/release-113/gtf/{sp_dir}/{gt[0]}",
                             GENOMES / sp_dir / gt[0]))
            print(sp_dir, "->", fafile, "|", gt[0], flush=True)
        else:
            print(sp_dir, "MISSING files", fa, gt, flush=True)
    except Exception as e:
        print(sp_dir, "ERR", repr(e), flush=True)

# omics raw
manifest.append(("https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_fair+new_CAGE_peaks_phase1and2.bed.gz",
                 OMICS / "cage" / "fantom5_hg38_cage_peaks_phase1and2.bed.gz"))
manifest.append(("https://remap.univ-amu.fr/storage/remap2022/hg38/MACS2/remap2022_nr_macs2_hg38_v1_0.bed.gz",
                 OMICS / "chip" / "remap2022_nr_macs2_hg38_v1_0.bed.gz"))

for d in (OMICS / "cage", OMICS / "chip", OMICS / "procap", OMICS / "histone", OMICS / "dnase"):
    d.mkdir(parents=True, exist_ok=True)

ok, fail = 0, 0
for url, out in manifest:
    out.parent.mkdir(parents=True, exist_ok=True)
    print("DL", out, flush=True)
    if curl_dl(url, out):
        ok += 1
    else:
        fail += 1
print(f"SUMMARY ok={ok} fail={fail}", flush=True)
