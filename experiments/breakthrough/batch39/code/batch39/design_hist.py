import gzip, re, collections
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
bins=collections.Counter()
n=0
with gzip.open(p,'rt',encoding='utf-8') as f:
    for line in f:
        if line.startswith('>'):
            m=re.match(r'^>chr17:(\d+)-(\d+)$',line.strip())
            if m:
                s=int(m.group(1)); bins[s//100000*100000]+=1; n+=1
print('n',n,'nbins',len(bins))
for k in sorted(bins):
    print(k, bins[k])
