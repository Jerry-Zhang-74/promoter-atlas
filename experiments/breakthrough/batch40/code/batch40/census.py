import gzip, re, json, collections, hashlib
from pathlib import Path
p = 'data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz'

hdr_re = re.compile(r'^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$')
recs = []
nonseq = 0
with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f:
    name = None; seqchunks = []
    for line in f:
        if line.startswith('>'):
            if name is not None:
                recs.append((name, ''.join(seqchunks)))
            name = line[1:].rstrip('\n')
            seqchunks = []
        else:
            seqchunks.append(line.strip())
    if name is not None:
        recs.append((name, ''.join(seqchunks)))

N = len(recs)
hdrs = [r[0] for r in recs]
seqs = [r[1] for r in recs]
hdr_mismatch = [h for h in hdrs if not hdr_re.match(h)]
seqlens = collections.Counter(len(s) for s in seqs)
dup_hdr = [k for k, c in collections.Counter(hdrs).items() if c > 1]
dup_seq = [k for k, c in collections.Counter(seqs).items() if c > 1]
# adapter presence
FWD = 'AGGACCGGATCAACT'; REV = 'CATTGCGTGAACCGA'
n_fwd_start = sum(1 for s in seqs if s.startswith(FWD))
n_rev_end = sum(1 for s in seqs if s.endswith(REV))
n_fwd_any = sum(1 for s in seqs if FWD in s)
n_rev_any = sum(1 for s in seqs if REV in s)
# parse headers
chroms = collections.Counter(); strands = collections.Counter()
parsed = []
for h in hdrs:
    m = hdr_re.match(h)
    if m:
        chroms[m.group('chr')] += 1; strands[m.group('strand')] += 1
        parsed.append((m.group('name'), m.group('chr'), int(m.group('start')), int(m.group('end')), m.group('strand')))
        # span check
n_span270 = sum(1 for _, _, s, e, _ in parsed if e - s == 270)
n_span_other = collections.Counter(e - s for _, _, s, e, _ in parsed if e - s != 270)

out = {
 'records': N,
 'header_pattern_mismatch': len(hdr_mismatch),
 'header_mismatch_examples': hdr_mismatch[:5],
 'seq_length_dist': dict(sorted(seqlens.items())),
 'dup_headers': len(dup_hdr), 'dup_header_examples': dup_hdr[:5],
 'dup_sequences': len(dup_seq), 'dup_seq_examples': [s[:30] for s in dup_seq[:3]],
 'adapter_fwd_start': n_fwd_start, 'adapter_rev_end': n_rev_end,
 'adapter_fwd_anywhere': n_fwd_any, 'adapter_rev_anywhere': n_rev_any,
 'chroms': dict(chroms.most_common()),
 'strands': dict(strands),
 'span270_count': n_span270, 'span_other': dict(n_span_other),
 'first3_headers': hdrs[:3],
 'last3_headers': hdrs[-3:],
}
json.dump(out, open('data/breakthrough_20261002/batch40/census.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
