"""Fetch optional upstream raw inputs and verify the recorded content hashes."""
import argparse,gzip,hashlib,json,shutil,urllib.request
from pathlib import Path
from promoter_atlas.layout import canonical_path
HERE=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target',default='.')
    ap.add_argument('--contains',required=True,help='Path substring, e.g. GTEx_, legacy_data/, or GRCh37; use ALL explicitly for all raw sources')
    ap.add_argument('--proxy')
    a=ap.parse_args();root=Path(a.target).resolve();root.mkdir(parents=True,exist_ok=True)
    rows=json.loads((HERE/'artifacts/manifests/UPSTREAM_DATA.json').read_text('utf-8'))['files']
    rows=[r for r in rows if a.contains=='ALL' or a.contains in r['path']]
    if not rows:raise SystemExit('No matching upstream files')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':a.proxy})) if a.proxy else urllib.request.build_opener()
    for r in rows:
        dest=canonical_path(root,r['path'])
        if not dest.is_relative_to(root):raise ValueError('Unsafe path')
        if dest.exists():
            if dest.stat().st_size==r['bytes'] and digest(dest)==r['sha256']:print('Verified',r['path']);continue
            raise FileExistsError('Existing upstream file differs: '+str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_name(dest.name+'.download')
        try:
            req=urllib.request.Request(r['url'],headers={'User-Agent':'PromoterAtlas-source/0.2.0'})
            with opener.open(req,timeout=300) as remote,tmp.open('wb') as out:
                if r.get('transport_compression')=='gzip':
                    with gzip.GzipFile(fileobj=remote) as uncompressed:shutil.copyfileobj(uncompressed,out,4*1024*1024)
                else:shutil.copyfileobj(remote,out,4*1024*1024)
            if tmp.stat().st_size!=r['bytes'] or digest(tmp)!=r['sha256']:raise ValueError('Upstream content hash mismatch: '+r['path'])
            tmp.replace(dest)
        finally:
            if tmp.exists():tmp.unlink()
        print('Downloaded and verified',r['path'],flush=True)
    print('Genomic .fa.gz/.gtf.gz files remain compressed unless transport_compression=gzip. Decompress them to the original names before running historical sequence builders.')

if __name__=='__main__':main()
