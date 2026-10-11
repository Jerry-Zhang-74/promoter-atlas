import gzip,sys,json,random,hashlib
from collections import defaultdict
from pyfaidx import Fasta
shard=sys.argv[1]; fa_path=sys.argv[2]; out=sys.argv[3]
random.seed(4404)
recs=[]
with gzip.open(shard,'rt',encoding='utf-8') as f:
    hdr=f.readline().rstrip('\n').split('\t'); ix={c:i for i,c in enumerate(hdr)}
    rp1=ix['Log2RPM_HAP1']; rp2=ix['Log2RPM_AGS']
    for line in f:
        c=line.rstrip('\n').split('\t')
        recs.append((c[ix['chr']],int(c[ix['start']]),int(c[ix['end']]),c[ix['strand']],c[rp1],c[rp2]))
g=defaultdict(list)
for ch,s,e,st,r1,r2 in recs:
    g[(ch,st,(e if st=='+' else s))].append((s,e,r1,r2))
elig=[(k,v) for k,v in g.items() if len(set(e-s for s,e,_,_ in v))>1 and len([1 for x in set(e-s for s,e,_,_ in v) if 100<=x<=600])>=2]
random.shuffle(elig)
fa=Fasta(fa_path,as_raw=False,duplicate_action='first')
comp=str.maketrans('ACGT','TGCA')
def rc(x): return x.translate(comp)[::-1]
def seq(ch,s,e,st):
    # 0-based half-open
    x=str(fa[ch][s:e]).upper()
    return rc(x) if st=='-' else x
checks=[]; n=0
strata={'plus':0,'minus':0,'long_short':0,'multi':0,'chroms':set()}
for k,v in elig:
    ch,st,end=k
    dls=sorted(set(e-s for s,e,_,_ in v)); dls=[x for x in dls if 100<=x<=600]
    if len(dls)<2: continue
    short_len,long_len=dls[0],dls[-1]
    bylen={e-s:(s,e,r1,r2) for s,e,r1,r2 in v}
    if short_len not in bylen or long_len not in bylen: continue
    if len(dls)>2 and strata['multi']<3 and n>=6:
        pass
    s_short=(bylen[short_len]); s_long=(bylen[long_len])
    # verify same oriented 3' end
    same_end = (s_short[1]==s_long[1]) if st=='+' else (s_short[0]==s_long[0])
    try:
        a=seq(ch,s_short[0],s_short[1],st); b=seq(ch,s_long[0],s_long[1],st)
    except KeyError as ex:
        continue
    is_suffix = (short_len<long_len and a==b[-len(a):])
    checks.append({"key":"%s|%s|%d"%(ch,st,end),"strand":st,"short_len":short_len,"long_len":long_len,
                   "same_3prime_end":bool(same_end),"short_is_exact_suffix":bool(is_suffix),
                   "short_sha256":hashlib.sha256(a.encode()).hexdigest()[:16],
                   "label_HAP1":s_short[2],"label_AGS":s_short[3],
                   "label_present_both": (s_short[2] not in ('','NA','nan') and s_long[2] not in ('','NA','nan'))})
    strata['plus' if st=='+' else 'minus']+=1
    strata['chroms'].add(ch)
    if short_len<=150 and long_len>=350: strata['long_short']+=1
    if len(dls)>2: strata['multi']+=1
    n+=1
    if n>=20: break
# ensure strand balance by continuing if needed
res={"n_checks":len(checks),"all_same_3prime_end":all(c['same_3prime_end'] for c in checks),
     "all_short_exact_suffix":all(c['short_is_exact_suffix'] for c in checks),
     "all_both_labels_present":all(c['label_present_both'] for c in checks),
     "strata":{k:(sorted(v) if isinstance(v,set) else v) for k,v in strata.items()},
     "pass_fraction": sum(1 for c in checks if c['same_3prime_end'] and c['short_is_exact_suffix'] and c['label_present_both'])/max(1,len(checks)),
     "checks":checks,
     "assembly":"GRCh38.primary_assembly.genome.fa (reference reconstruction at author-measured boundaries; actual donor HG02601 DNA not distributed)",
     "note":"HAP1/AGS shard; structural DNA-suffix check only. Because both fragments share the same 3' boundary and the DNA is read from one reference, suffix identity is arithmetically expected; this verifies boundary parsing, not independent sequence identity."}
open(out,'w',encoding='utf-8').write(json.dumps(res,indent=2,ensure_ascii=True))
print(json.dumps({k:v for k,v in res.items() if k!='checks'},indent=2))
print("sample checks:",json.dumps(res['checks'][:4],indent=1))
