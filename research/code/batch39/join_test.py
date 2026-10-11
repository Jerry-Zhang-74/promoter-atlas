import gzip, json, re, collections
s5=json.load(open('data/breakthrough_20261002/batch39/s5_rows_raw.json',encoding='utf-8'))
pat=re.compile(r'::(chr[0-9XYM]+):(\d+)-(\d+)([+-])?$')
real=collections.defaultdict(set)   # chrom -> set((start,end))
for r in s5:
    m=pat.search(r['A'])
    if m: real[m.group(1)].add((int(m.group(2)),int(m.group(3))))
print('s5 chroms',{k:len(v) for k,v in sorted(real.items())})
target=real.get('chr17',set())
tstart=collections.defaultdict(list)
for s,e in target: tstart[s].append(e)
print('chr17 s5 coords',len(target))
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
exact=0; start_hit=0; cover_hit=0; n=0
cover_starts=set(tstart.keys())
seen_records=0
# To test 'covering': need windows interval overlap. Load all chr17 windows (start,end) - 3.9M, store compactly
wins=[]
with gzip.open(p,'rt',encoding='utf-8') as f:
    for line in f:
        if line.startswith('>'):
            m=re.match(r'^>chr17:(\d+)-(\d+)$',line.strip())
            if m:
                s=int(m.group(1)); e=int(m.group(2)); wins.append((s,e))
print('chr17 windows',len(wins))
wset=set(wins)
for s,e in target:
    if (s,e) in wset: exact+=1
# covering: does any window start<=s and end>=e
wins_sorted=sorted(wins)
import bisect
ws=[w[0] for w in wins_sorted]
mx=[]  # prefix max end up to i
run=-1
prefix_max=[]
for s,e in wins_sorted:
    run=max(run,e); prefix_max.append(run)
def covered(s,e):
    i=bisect.bisect_right(ws,s)-1
    return i>=0 and prefix_max[i]>=e
cov=sum(1 for s,e in target if covered(s,e))
print('exact_header_matches',exact,'covered_by_some_window',cov,'of',len(target))
json.dump({'chr17_s5':len(target),'exact_header_matches':exact,'covered_by_window':cov,
           'design_windows':len(wins)}, open('data/breakthrough_20261002/batch39/join_design.json','w'),indent=1)
