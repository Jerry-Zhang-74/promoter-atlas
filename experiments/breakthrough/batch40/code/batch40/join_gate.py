import gzip, re, json, collections
p = 'data/breakthrough_20261002/batch40/source/GSM9580234_design_rmIllegalChars.fa.gz'
s5 = json.load(open('data/breakthrough_20261002/batch39/s5_rows_raw.json', encoding='utf-8'))
hdr_re = re.compile(r'^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$')

# --- load design headers ---
design_hdr = []   # full header string
design_seq = {}
name_to_hdrs = collections.defaultdict(list)
coord_to_hdrs = collections.defaultdict(list)
with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f:
    name = None; chunks = []
    def flush(nm, sq):
        design_hdr.append(nm); design_seq[nm] = sq
        m = hdr_re.match(nm)
        if m:
            name_to_hdrs[m.group('name')].append(nm)
            coord_to_hdrs[(m.group('chr'), int(m.group('start')), int(m.group('end')), m.group('strand'))].append(nm)
    for line in f:
        if line.startswith('>'):
            if name is not None: flush(name, ''.join(chunks))
            name = line[1:].rstrip('\n'); chunks = []
        else:
            chunks.append(line.strip())
    if name is not None: flush(name, ''.join(chunks))

design_set = set(design_hdr)
design_dup_hdr = [k for k, c in collections.Counter(design_hdr).items() if c > 1]

# --- classify s5 ---
real, ctrl = [], []
for r in s5:
    (real if r['F'] == '0' else ctrl).append(r)

def match_label(lab):
    return lab in design_set

real_matched = [r for r in real if match_label(r['A'])]
real_unmatched = [r for r in real if not match_label(r['A'])]
ctrl_matched = [r for r in ctrl if match_label(r['A'])]
ctrl_unmatched = [r for r in ctrl if not match_label(r['A'])]

# unique-binding check: a matched label maps to exactly one design record if design has no dup header
def unique_ok(lab):
    return lab in design_set  # design_dup_hdr==0 -> unique

# conflict analysis for unmatched: coordinate overlap (ignoring name)
unmatched_coord_hits = 0
unmatched_coord_examples = []
for r in real_unmatched:
    m = hdr_re.search(r['A'])
    if m:
        key = (m.group('chr'), int(m.group('start')), int(m.group('end')), m.group('strand'))
        if coord_to_hdrs.get(key):
            unmatched_coord_hits += 1
            if len(unmatched_coord_examples) < 5:
                unmatched_coord_examples.append({'s5': r['A'], 'design_hdr_same_coord': coord_to_hdrs[key]})

# name-only (strip _index_+) match for unmatched to see if same family present
unmatched_name_hits = 0
unmatched_name_examples = []
for r in real_unmatched:
    m = hdr_re.search(r['A'])
    if m and m.group('name') in name_to_hdrs:
        unmatched_name_hits += 1
        if len(unmatched_name_examples) < 5:
            unmatched_name_examples.append({'s5': r['A'], 'design_same_name': name_to_hdrs[m.group('name')][:3]})

# control identity check: do s5 scrambled labels look like design scrambled headers?
ctrl_scram_labels = [r['A'] for r in ctrl][:10]
design_scram_headers = [h for h in design_hdr if h.startswith('scrambled')]

out = {
 'design_records': len(design_hdr),
 'design_coord_headers': sum(1 for h in design_hdr if hdr_re.match(h)),
 'design_scrambled_headers': len(design_scram_headers),
 'design_dup_headers': len(design_dup_hdr),
 's5_total': len(s5), 's5_real': len(real), 's5_controls': len(ctrl),
 'real_matched': len(real_matched), 'real_unmatched': len(real_unmatched),
 'real_match_fraction': round(len(real_matched)/len(real), 6),
 'ctrl_matched': len(ctrl_matched), 'ctrl_unmatched': len(ctrl_unmatched),
 'unique_binding_all_matched': all(unique_ok(r['A']) for r in real_matched),
 'unmatched_coord_overlap_hits': unmatched_coord_hits,
 'unmatched_coord_examples': unmatched_coord_examples,
 'unmatched_name_family_hits': unmatched_name_hits,
 'unmatched_name_examples': unmatched_name_examples,
 'ctrl_scram_label_examples': ctrl_scram_labels,
 'design_scram_header_examples': design_scram_headers[:10],
 'ctrl_label_in_design_scram': sum(1 for r in ctrl if r['A'] in design_set),
}
json.dump(out, open('data/breakthrough_20261002/batch40/join_gate.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
# save matched design seq map for later (only matched oligos)
