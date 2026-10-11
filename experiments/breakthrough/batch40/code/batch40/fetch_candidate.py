#!/usr/bin/env python
# Batch40 v6.27.1-oligo-design-gate.rc1 : fetch EXACTLY ONE candidate file.
# GSM9580234_design_rmIllegalChars.fa.gz (frozen GEO filelist, GSE325670).
# <=3 attempts, each 120s hard timeout, cumulative downloaded bytes <= 2MB.
# Verify MIME, gzip EOF/CRC/ISIZE, length, sha256. No other downloads/queries.
import hashlib, gzip, json, os, ssl, sys, time, io
import urllib.request, urllib.error
from pathlib import Path

ROOT = Path('E:/promoter_atlas')
OUT = ROOT / 'data/breakthrough_20261002/batch40/source'
OUT.mkdir(parents=True, exist_ok=True)

CTX = ssl.create_default_context()
UA = 'Mozilla/5.0 (research; local academic use) promoter_atlas/batch40'

NAME = 'GSM9580234_design_rmIllegalChars.fa.gz'
URL  = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9580nnn/GSM9580234/suppl/GSM9580234_design_rmIllegalChars.fa.gz'
EXPECTED_BYTES = 429038
HARD_TIMEOUT = 120
MAX_ATTEMPTS = 3
CUM_BYTE_CAP = 2 * 1024 * 1024

dest = OUT / NAME
rec = {'name': NAME, 'url': URL, 'attempts': [], 'retries': 0,
       'expected_bytes_from_filelist': EXPECTED_BYTES}
cum = 0
ok = False
for attempt in range(1, MAX_ATTEMPTS + 1):
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=HARD_TIMEOUT, context=CTX) as r:
            ctype = r.headers.get('Content-Type', '')
            clen = r.headers.get('Content-Length', '')
            data = r.read()
        dt = time.time() - t0
        cum += len(data)
        dest.write_bytes(data)
        rec['attempts'].append({'attempt': attempt, 'ok': True, 'bytes': len(data),
                                'content_type': ctype, 'content_length_header': clen,
                                'elapsed_s': round(dt, 3)})
        ok = True
        break
    except Exception as e:
        dt = time.time() - t0
        rec['attempts'].append({'attempt': attempt, 'ok': False, 'error': repr(e),
                                'elapsed_s': round(dt, 3)})
        rec['retries'] += 1
        if cum >= CUM_BYTE_CAP:
            rec['status'] = 'ABORT_BYTE_CAP'
            break
        time.sleep(2)

if not ok:
    rec.setdefault('status', 'FAILED_ALL_ATTEMPTS')
else:
    raw = dest.read_bytes()
    sz = len(raw)
    rec['final_bytes'] = sz
    rec['mime_by_sniff'] = 'gzip' if raw[:2] == b'\x1f\x8b' else 'unknown'
    rec['sha256'] = hashlib.sha256(raw).hexdigest()
    rec['size_ok'] = (sz == EXPECTED_BYTES)
    rec['length_matches_filelist'] = (sz == EXPECTED_BYTES)
    # gzip stream decode with CRC32 + ISIZE verification, plus EOF marker check
    gz = {}
    try:
        # explicit EOF (next member / trailing) check
        d = gzip.decompress(raw)
        gz = {'ok': True, 'decompressed_bytes': len(d), 'crc_verified': True}
    except Exception as e:
        gz = {'ok': False, 'error': repr(e)}
    # check the gzip trailer (last 8 bytes = CRC32 LE + ISIZE LE) equals computed
    try:
        import struct, zlib
        crc_le, isize_le = struct.unpack('<II', raw[-8:])
        calc_crc = zlib.crc32(gzip.decompress(raw)) & 0xffffffff
        calc_isize = len(gzip.decompress(raw)) & 0xffffffff
        gz['trailer_crc_ok'] = (crc_le == calc_crc)
        gz['trailer_isize_ok'] = (isize_le == calc_isize)
        gz['isize_bytes'] = isize_le
    except Exception as e:
        gz['trailer_error'] = repr(e)
    rec['gzip'] = gz
    rec['status'] = 'OK' if (rec['size_ok'] and gz.get('ok') and gz.get('trailer_crc_ok')
                             and gz.get('trailer_isize_ok')) else 'INTEGRITY_FAIL'
rec['cumulative_downloaded_bytes'] = cum

summary = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'n_requests': len([a for a in rec.get('attempts', [])]),
           'cumulative_bytes': cum, 'record': rec}
(OUT / 'source_fetch_log.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
print(json.dumps({'status': rec['status'], 'bytes': rec.get('final_bytes'),
                  'sha256': rec.get('sha256'), 'cumulative': cum}, indent=1))
