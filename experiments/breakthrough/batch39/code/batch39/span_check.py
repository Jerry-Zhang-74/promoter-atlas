import gzip, re, collections
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
spans=collections.Counter(); seqlens=collections.Counter(); n=0
with gzip.open(p,'rt',encoding='utf-8') as f:
    name=None
    for line in f:
        if line.startswith('>'):
            m=re.match(r'^>chr17:(\d+)-(\d+)$',line.strip())
            if m:
                spans[int(m.group(2))-int(m.group(1))]+=1
        else:
            seqlens[len(line.strip())]+=1
print('span==269/270/271/272:', {k:spans.get(k,0) for k in (268,269,270,271,272)})
print('span_min_max', min(spans), max(spans))
print('top spans', spans.most_common(10))
print('seq len ==270/271:', {k:seqlens.get(k,0) for k in (270,271)})
print('total seqs', sum(seqlens.values()))
