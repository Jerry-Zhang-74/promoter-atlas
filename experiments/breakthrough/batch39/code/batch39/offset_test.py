import gzip, json, re, collections
s5=json.load(open('data/breakthrough_20261002/batch39/s5_rows_raw.json',encoding='utf-8'))
pat=re.compile(r'::(chr17):(\d+)-(\d+)([+-])?$')
tgt=[(int(m.group(2)),int(m.group(3))) for r in s5 if (m:=pat.search(r['A']))]
print('s5 chr17 coords',len(tgt))
ws=set()
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
with gzip.open(p,'rt',encoding='utf-8') as f:
    for line in f:
        if line.startswith('>'):
            m=re.match(r'^>chr17:(\d+)-(\d+)$',line.strip())
            if m: ws.add((int(m.group(1)),int(m.group(2))))
print('windows',len(ws))
for ds in (-271,-270,-1,0,1):
  for de in (-271,-270,-1,0,1):
    c=sum(1 for s,e in tgt if (s+ds,e+de) in ws)
    if c>0: print('offset',ds,de,'matches',c)
# also: does any window contain the s5 start as its start?
starts={s for s,e in ws}
hit=sum(1 for s,e in tgt if s in starts)
print('s5_start_in_window_starts',hit)
