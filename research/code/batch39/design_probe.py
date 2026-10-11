import gzip, re, collections
p='data/breakthrough_20261002/batch39/source/GSM9609016_designfile.fa.gz'
targets=['chr17:45870111-45870381','chr17:46764547-46764817','chr17:44481424-44481694']
hdr_pat=re.compile(r'^>(\S+)')
starts=[]; ends=[]; strands=collections.Counter(); nlen=collections.Counter()
n=0; hits={t:0 for t in targets}; sample_hits={}
maxlen=0
with gzip.open(p,'rt',encoding='utf-8') as f:
    for line in f:
        if line.startswith('>'):
            n+=1
            h=line[1:].strip()
            m=re.match(r'^(chr[0-9XYM]+):(\d+)-(\d+)(.*)$',h)
            if m:
                s=int(m.group(2)); e=int(m.group(3)); strand=m.group(4)
                starts.append(s); ends.append(e); strands[strand]+=1
            else:
                strands['NONCOORD:'+h[:20]]+=1
        else:
            L=len(line.strip())
            if L>maxlen: maxlen=L
n_s=len(starts)
print('total_records',n,'coord_headers',n_s)
if starts:
    print('start_range',min(starts),max(starts))
    print('end_range',min(ends),max(ends))
    print('strand_field_top',strands.most_common(6))
    span=collections.Counter((e-s) for s,e in zip(starts,ends))
    print('span_top',span.most_common(8))
print('max_seqlen',maxlen)
