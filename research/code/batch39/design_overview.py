import gzip, json, collections
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
n=0; lens=collections.Counter(); names=[]; head_prefix=collections.Counter()
srclens=collections.Counter()
first=[]
with gzip.open(p,'rt',encoding='utf-8') as f:
    name=None; seq=[]
    def flush():
        global name,seq
        if name is not None:
            s=''.join(seq); lens[len(s)]+=1
            if len(first)<6: first.append((name,s[:40],len(s)))
        name=None; seq=[]
    for line in f:
        if line.startswith('>'):
            flush(); name=line[1:].strip(); n+=1
        else:
            seq.append(line.strip())
    flush()
print('n_records',n)
print('len_dist',lens.most_common(10))
print('first6',first)
# header patterns
import re
with gzip.open(p,'rt',encoding='utf-8') as f:
    c=0
    for line in f:
        if line.startswith('>'):
            h=line[1:].strip()
            head_prefix[h[:14]]+=1
            if c<8: print('HDR',h)
            c+=1
        if c>=4000: break
print('prefix_top',head_prefix.most_common(8))
