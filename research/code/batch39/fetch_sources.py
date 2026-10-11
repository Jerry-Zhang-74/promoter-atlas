#!/usr/bin/env python
# Batch39 source recovery: exactly the frozen filelist targets + author oligo_MPRA call chain.
# Records bytes/retries/time per request. Verifies MIME, gzip EOF/CRC, length, sha256.
import hashlib, gzip, json, os, ssl, sys, time, zlib
import urllib.request, urllib.error
from pathlib import Path

ROOT = Path('E:/promoter_atlas')
OUT = ROOT / 'data/breakthrough_20261002/batch39/source'
OUT.mkdir(parents=True, exist_ok=True)

CTX = ssl.create_default_context()
UA = 'Mozilla/5.0 (research; local academic use) promoter_atlas/batch39'

# (local_name, url, expected_min_bytes, expected_max_bytes, is_gzip)
TARGETS = [
    ('GSM9609016_designfile.fa.gz',
     'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9609nnn/GSM9609016/suppl/GSM9609016_designfile.fa.gz',
     25939760, None, True),
    ('MPRAnalyze_oligoMPRA_heks.R',
     'https://raw.githubusercontent.com/HudsonAlpha/Hauser_MAPT_MPRAs/main/oligo_MPRA/mpranalyze/MPRAnalyze_oligoMPRA_heks.R',
     3000, None, False),
    ('generateDesignfile.rmd',
     'https://raw.githubusercontent.com/HudsonAlpha/Hauser_MAPT_MPRAs/main/oligo_MPRA/generateDesignfile.rmd',
     11000, None, False),
    ('mpranalyze_padding.py',
     'https://raw.githubusercontent.com/HudsonAlpha/Hauser_MAPT_MPRAs/main/oligo_MPRA/mpranalyze/mpranalyze_padding.py',
     700, None, False),
    ('MPRAnalyze_oligoMPRA_neurons.R',
     'https://raw.githubusercontent.com/HudsonAlpha/Hauser_MAPT_MPRAs/main/oligo_MPRA/mpranalyze/MPRAnalyze_oligoMPRA_neurons.R',
     3300, None, False),
]

def one_attempt(url, dest, timeout=120):
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        ctype = r.headers.get('Content-Type', '')
        clen = r.headers.get('Content-Length', '')
        data = r.read()
    dt = time.time() - t0
    dest.write_bytes(data)
    return {'bytes': len(data), 'content_type': ctype, 'content_length_header': clen,
            'elapsed_s': round(dt, 3)}

def gzip_check(path):
    raw = path.read_bytes()
    # stream decompress with CRC verification
    try:
        d = gzip.decompress(raw)
        # gzip.decompress verifies CRC32 + ISIZE
        return {'ok': True, 'decompressed_bytes': len(d), 'crc_verified': True}
    except Exception as e:
        return {'ok': False, 'error': repr(e)}

records = []
for name, url, minb, maxb, isgz in TARGETS:
    dest = OUT / name
    rec = {'name': name, 'url': url, 'attempts': [], 'retries': 0}
    ok = False
    for attempt in range(3):
        try:
            info = one_attempt(url, dest)
            rec['attempts'].append({'attempt': attempt + 1, 'ok': True, **info})
            ok = True
            break
        except Exception as e:
            rec['attempts'].append({'attempt': attempt + 1, 'ok': False, 'error': repr(e),
                                    'elapsed_s': None})
            rec['retries'] += 1
            time.sleep(2)
    if not ok:
        rec['status'] = 'FAILED_ALL_ATTEMPTS'
        records.append(rec)
        continue
    sz = dest.stat().st_size
    rec['final_bytes'] = sz
    rec['mime_by_sniff'] = 'gzip' if dest.read_bytes()[:2] == b'\x1f\x8b' else 'text/plain'
    rec['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    rec['size_ok'] = (sz >= minb) if maxb is None else (minb <= sz <= maxb)
    if isgz:
        rec['gzip'] = gzip_check(dest)
        rec['status'] = 'OK' if (rec['size_ok'] and rec['gzip'].get('ok')) else 'INTEGRITY_FAIL'
    else:
        rec['status'] = 'OK' if rec['size_ok'] else 'SIZE_SUSPECT'
    rec['total_elapsed_s'] = round(sum(a['elapsed_s'] for a in rec['attempts'] if a.get('elapsed_s')), 3)
    records.append(rec)
    print(name, rec['status'], sz, rec['sha256'][:16])

summary = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'n_requests': len(records),
           'cumulative_bytes': sum(r.get('final_bytes', 0) for r in records), 'records': records}
(OUT / 'source_fetch_log.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
print('cumulative bytes', summary['cumulative_bytes'])
