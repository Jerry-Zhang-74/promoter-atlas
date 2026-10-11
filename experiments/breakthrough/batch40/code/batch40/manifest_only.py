import json, hashlib, time
from pathlib import Path
ROOT=Path('E:/promoter_atlas'); R=ROOT/'reports/breakthrough_20261002/batch40'
GEN=time.strftime('%Y-%m-%dT%H:%M:%S%z')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.load(open(R/'sha_manifest.json',encoding='utf-8')); m['generated_at']=GEN
for f,v in m['files'].items():
    p=ROOT/f
    m['files'][f]={'sha256':sha(p),'bytes':p.stat().st_size} if p.exists() else {'sha256':None,'reason':'missing'}
json.dump(m,open(R/'sha_manifest.json','w',encoding='utf-8'),indent=1)
miss=[f for f,v in m['files'].items() if v.get('sha256') is None]
print('files',len(m['files']),'missing',miss)
