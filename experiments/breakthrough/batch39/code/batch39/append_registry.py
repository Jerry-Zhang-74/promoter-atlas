import json,os,time
from pathlib import Path
ROOT=Path('E:/promoter_atlas')
reg=ROOT/'reports/breakthrough_20261002/version_registry.json'
ent=json.load(open(ROOT/'reports/breakthrough_20261002/batch39/version_registry_entry.json',encoding='utf-8'))
o=json.load(open(reg,encoding='utf-8'))
o.setdefault('entries',[])
if not any(e.get('batch')=='batch39' for e in o['entries']):
    o['entries'].append(ent)
o['last_appended_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
o['last_appended_by']='batch39 worker (DeepSeek Flash implementer; parent-authorized append-only)'
tmp=reg.with_suffix('.json.tmp'); tmp.write_text(json.dumps(o,indent=1,ensure_ascii=False),encoding='utf-8'); os.replace(tmp,reg)
ch=ROOT/'reports/breakthrough_20261002/CHANGELOG.txt'
with open(ch,'a',encoding='utf-8') as f:
    f.write('%s batch39 v6.27.0-hekft-fragment-readiness.rc1: HEK293FT OligoMPRA data readiness -> NOT_READY; authorized GSM9609016_designfile is a chr17 Barcode-Association reference (join 1/5868), measured design DNA missing; 5 sources 25,959,333B; zero fit/GPU; minP=UNKNOWN; b37 preflight FAIL and local293FT D28 FAIL preserved.\n'%time.strftime('%Y-%m-%dT%H:%M:%S%z'))
print('appended; entries',len(o['entries']))
