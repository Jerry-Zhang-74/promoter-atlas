"""Download the fixed public release assets and verify original SHA-256 values."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
BASE_URL = 'https://github.com/Jerry-Zhang-74/promoter-atlas/releases/download/v0.1.0'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proxy', help='Optional HTTPS proxy, e.g. http://127.0.0.1:7897')
    args = ap.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': args.proxy})) if args.proxy else urllib.request.build_opener()
    manifest = json.loads((HERE / 'WEIGHTS_MANIFEST.json').read_text('utf-8'))
    for part in manifest['parts']:
        name = part['file']
        if Path(name).name != name:
            raise ValueError('Manifest asset must be a simple filename')
        target = HERE / name
        if target.exists() and target.stat().st_size == part['bytes'] and digest(target) == part['sha256']:
            print('Verified existing', name)
            continue
        temporary = target.with_suffix('.pt.part')
        request = urllib.request.Request(BASE_URL + '/' + name, headers={'User-Agent':'PromoterAtlas/0.1.0'})
        try:
            with opener.open(request, timeout=120) as response, temporary.open('wb') as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if temporary.stat().st_size != part['bytes'] or digest(temporary) != part['sha256']:
                raise ValueError('Weight checksum/size mismatch: ' + name)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        print('Downloaded and verified', name)

if __name__ == '__main__':
    main()
