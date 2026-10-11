# -*- coding: utf-8 -*-
# batch52 v6.38.0-full-hek-nested-multitask.rc1 -- FULL-shard nested cohort builder.
# NO batch51 500-component/100k-endpoint cap. Author GSE301246 v2_MH4 fold0, HEK293 Log2Norm_HEK293.
# Reference: official Ensembl GRCh37 primary assembly (== hg19 assembly). ZERO fits.
import gzip, hashlib, json, time, collections, os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_fasta import Faidx

ROOT = r"E:/promoter_atlas"
DATA = os.path.join(ROOT, "data/breakthrough_20261002/batch52")
B51  = os.path.join(ROOT, "data/breakthrough_20261002/batch51")
BED  = os.path.join(B51, "GSE301246_PARM_Training_Library_v2_MH4_normalized_fold0.bed.gz")
FA   = os.path.join(DATA, "reference/GRCh37.primary_assembly.fa")
OUT  = os.path.join(DATA, "cohort"); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time()-T0), *a, flush=True)
def h(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
COMP = bytes.maketrans(b'ACGTNacgtn', b'TGCANtgcan')
def rc(s): return s.translate(COMP)[::-1]
MINL, MAXL = 100, 600
SALT_SPLIT = 'b52-full-hek-v1|'

rows = 0; R = []; bad_hek = 0
with gzip.open(BED, 'rt', encoding='utf-8') as f:
    header = f.readline().rstrip('\n').split('\t')
    for line in f:
        a = line.rstrip('\n').split('\t'); rows += 1
        try: hek = float(a[5])
        except Exception: hek = float('nan'); bad_hek += 1
        R.append((a[1], int(a[2]), int(a[3]), a[4], a[44], int(a[45]), int(a[46]), a[47], a[48], hek))
log('parsed', rows, 'rows; hek NA', bad_hek, 'header_cols', len(header))

elig = [i for i, r in enumerate(R) if MINL <= (r[2]-r[1]) <= MAXL]
lengths = sorted(R[i][2]-R[i][1] for i in elig)
log('eligible', len(elig), 'len min/med/max', lengths[0], lengths[len(lengths)//2], lengths[-1])

by_coord = collections.defaultdict(list)
for i in elig:
    r = R[i]; by_coord['%s:%d-%d:%s' % (r[0], r[1], r[2], r[3])].append(i)
n_dup_exact = 0; n_dup_conflict = 0
for cid, idxs in by_coord.items():
    if len(idxs) > 1:
        vals = set(R[i][9] for i in idxs)
        if len(vals) == 1: n_dup_exact += 1
        else: n_dup_conflict += 1
log('coord_id duplicates exact', n_dup_exact, 'conflict', n_dup_conflict)

ep2rows = collections.defaultdict(list); row2ep = {}
for i in elig:
    r = R[i]; pos = r[2] if r[3] == '+' else r[1]
    k = (r[0], r[3], pos); ep2rows[k].append(i); row2ep[i] = k
log('distinct endpoints', len(ep2rows))

parent = list(range(len(R)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)
by_feat = collections.defaultdict(list)
for i in elig: by_feat[R[i][4]].append(i)
fe = 0
for fidx, idxs in by_feat.items():
    a0 = idxs[0]
    for b in idxs[1:]: union(a0, b); fe += 1
by_chr = collections.defaultdict(list)
for i in elig: by_chr[R[i][0]].append(i)
ov = 0
for ch, idxs in by_chr.items():
    idxs.sort(key=lambda i: (R[i][1], R[i][2]))
    cur = None; curmax = -1
    for i in idxs:
        s, e = R[i][1], R[i][2]
        if cur is None or s >= curmax:
            cur = i; curmax = e
        else:
            union(cur, i); ov += 1
            if e > curmax: curmax = e
log('FEAT union edges', fe, 'nFEAT', len(by_feat), 'FEATtype', dict(collections.Counter(R[i][8] for i in elig)), 'overlap edges', ov)

fx = Faidx(FA)
seqs = {}
n_amb_rows = 0; amb_bases = 0
for j, i in enumerate(elig):
    r = R[i]
    name = r[0][3:] if r[0].startswith('chr') else r[0]
    s = fx.fetch(name, r[1], r[2])
    if r[3] == '-': s = rc(s)
    if 'N' in s:
        n_amb_rows += 1; amb_bases += s.count('N')
    seqs[i] = s
    if (j+1) % 200000 == 0: log('seq extracted', j+1)
log('sequences extracted', len(seqs), 'rows_with_N', n_amb_rows, 'N_bases', amb_bases)
seqsha = {i: h(seqs[i]) for i in elig}

seqkey = {}
for i in elig:
    for key in (seqs[i], rc(seqs[i])):
        if key in seqkey:
            union(seqkey[key], i)
        else:
            seqkey[key] = i
log('exact/RC seq union done; distinct seq keys', len(seqkey))

def components():
    c = collections.defaultdict(list)
    for i in elig: c[find(i)].append(i)
    return list(c.values())
comps = components()
log('components after FEAT/overlap/seq union', len(comps))

def comp_rep(idxs):
    return max(idxs, key=lambda i: (len(seqs[i]), i))

homology = {"tool": "blastn 2.12.0", "task": "blastn", "identity_min": 90, "coverage_min": 90,
            "searched_fraction": "component representatives only (one longest seq per component)",
            "pairs_verified": 0, "matches_union": 0, "runtime_s": 0.0, "status": "skipped"}
try:
    tH = time.time()
    repdir = os.path.join(OUT, 'homology'); os.makedirs(repdir, exist_ok=True)
    repfa = os.path.join(repdir, 'reps.fa')
    repmap = {}
    with open(repfa, 'w', encoding='utf-8', newline='\n') as fo:
        for ci, idxs in enumerate(comps):
            i = comp_rep(idxs); repmap['c%d' % ci] = ci
            fo.write('>c%d\n%s\n' % (ci, seqs[i]))
    dbfa = os.path.join(repdir, 'reps_db')
    subprocess.run(['makeblastdb', '-in', repfa, '-dbtype', 'nucl', '-out', dbfa], check=True, capture_output=True)
    res = subprocess.run(['blastn', '-query', repfa, '-db', dbfa, '-task', 'blastn',
                          '-outfmt', '6 qseqid sseqid pident length qlen slen',
                          '-evalue', '1e-10', '-max_target_seqs', '200', '-num_threads', '4'],
                         check=True, capture_output=True, text=True)
    npair = 0; nmerge = 0; seen = set()
    for line in res.stdout.splitlines():
        f = line.split('\t'); q, s2 = f[0], f[1]
        if q == s2: continue
        key = (q, s2) if q < s2 else (s2, q)
        if key in seen: continue
        seen.add(key); npair += 1
        pident = float(f[2]); alnlen = int(f[3]); qlen = int(f[4]); slen = int(f[5])
        if pident >= 90 and alnlen/qlen*100 >= 90 and alnlen/slen*100 >= 90:
            a1 = comps[repmap[q]][0]; a2 = comps[repmap[s2]][0]
            if find(a1) != find(a2): union(a1, a2); nmerge += 1
    homology.update({"pairs_verified": npair, "matches_union": nmerge,
                     "runtime_s": round(time.time()-tH, 1), "status": "ok"})
    log('homology candidates', npair, 'unions', nmerge)
except Exception as e:
    homology.update({"status": "error", "error": str(e)[:300]})
    log('homology error', str(e)[:200])

comps = components()
log('components after homology union', len(comps))

def split_of(uid):
    v = int(h(SALT_SPLIT + uid)[:8], 16) % 100
    return 'TRAIN' if v < 70 else ('DEV' if v < 85 else 'TEST')

members = []; groups = []; pair_rows = []; geom_bad = 0; geom_ok = 0
for idxs in comps:
    uid = min(seqsha[i] for i in idxs); sp = split_of(uid)
    eps = collections.defaultdict(dict)
    for i in idxs:
        ep = row2ep[i]; L = R[i][2]-R[i][1]
        if L not in eps[ep]: eps[ep][L] = i
    n_ep = len(eps); n_pairs = sum(max(0, len(v)-1) for v in eps.values())
    groups.append((uid, sp, len(idxs), n_ep, n_pairs))
    for i in idxs:
        r = R[i]
        members.append(('%s:%d-%d:%s' % (r[0], r[1], r[2], r[3]), uid, sp, r[0], r[1], r[2], r[3],
                        r[2]-r[1], seqsha[i], r[4], r[7], r[8]))
    for ep, ld in eps.items():
        ls = sorted(ld.keys())
        for k in range(len(ls)-1):
            si = ld[ls[k]]; li = ld[ls[k+1]]
            ss = seqs[si]; sl = seqs[li]
            if ss == sl[-len(ss):]: geom_ok += 1
            else: geom_bad += 1
            pair_rows.append((uid, sp, ep[0], ep[1], ep[2],
                              '%s:%d-%d:%s' % (R[si][0], R[si][1], R[si][2], R[si][3]), ls[k],
                              '%s:%d-%d:%s' % (R[li][0], R[li][1], R[li][2], R[li][3]), ls[k+1]))
log('components', len(comps), 'members', len(members), 'pairs', len(pair_rows),
    'geom_ok', geom_ok, 'geom_bad', geom_bad)
cnt = collections.Counter(g[1] for g in groups)
cnt_p = collections.Counter(pr[1] for pr in pair_rows)
log('comp split counts', dict(cnt), 'pair split counts', dict(cnt_p))

with open(os.path.join(OUT, 'members.tsv'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('coord_id\tunion_id\tsplit\tchr\tstart\tend\tstrand\tlength\tseqsha\tFEAT\tFEATsense\tFEATtype\n')
    for m in members: f.write('\t'.join(str(x) for x in m) + '\n')
with open(os.path.join(OUT, 'groups.tsv'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('union_id\tsplit\tn_members\tn_endpoints\tn_adjacent_pairs\n')
    for g in groups: f.write('\t'.join(str(x) for x in g) + '\n')

cid2lab = {}
for i in elig:
    r = R[i]; cid2lab['%s:%d-%d:%s' % (r[0], r[1], r[2], r[3])] = r[9]
pair_rows.sort()
ntr = ndv = nte = 0
with open(os.path.join(OUT, 'pairs_train_dev.tsv'), 'w', encoding='utf-8', newline='\n') as trd, \
     open(os.path.join(DATA, 'SEALED_test_labels.tsv'), 'w', encoding='utf-8', newline='\n') as sealed:
    trd.write('union_id\tsplit\tchr\tstrand\tthree_prime_end\tshort_coord_id\tshort_len\tlong_coord_id\tlong_len\tshort_log2norm_hek\tlong_log2norm_hek\tdelta_short_minus_long\n')
    sealed.write('union_id\tshort_coord_id\tshort_len\tlong_coord_id\tlong_len\tshort_log2norm_hek\tlong_log2norm_hek\tdelta_short_minus_long\n')
    for pr in pair_rows:
        uid, sp, ch, st, pos, scid, sl, lcid, ll = pr
        slab = cid2lab.get(scid, float('nan')); llab = cid2lab.get(lcid, float('nan'))
        d = (slab - llab) if (slab == slab and llab == llab) else float('nan')
        if sp == 'TEST':
            sealed.write('%s\t%s\t%d\t%s\t%d\t%.6f\t%.6f\t%.6f\n' % (uid, scid, sl, lcid, ll, slab, llab, d)); nte += 1
        else:
            trd.write('%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\t%d\t%.6f\t%.6f\t%.6f\n' % (uid, sp, ch, st, pos, scid, sl, lcid, ll, slab, llab, d))
            if sp == 'TRAIN': ntr += 1
            else: ndv += 1
log('pairs TRAIN', ntr, 'DEV', ndv, 'TEST_sealed', nte)

with open(os.path.join(OUT, 'endpoint_seqs.tsv'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('coord_id\tchr\tstrand\tthree_prime_end\tlength\tseq\n')
    for i in elig:
        r = R[i]
        f.write('%s:%d-%d:%s\t%s\t%s\t%d\t%d\t%s\n' % (r[0], r[1], r[2], r[3], r[0], r[3],
                (r[2] if r[3] == '+' else r[1]), r[2]-r[1], seqs[i]))
log('endpoint_seqs written')

summary = {"code": "build_cohort_full.py", "source_bed": BED, "salt_split": SALT_SPLIT,
    "nrows_total": rows, "n_eligible": len(elig), "n_dup_exact_coord": n_dup_exact,
    "n_dup_conflict_coord": n_dup_conflict, "n_endpoints_total": len(ep2rows),
    "n_components_full": len(comps), "n_members": len(members), "n_pairs": len(pair_rows),
    "geom_ok": geom_ok, "geom_bad": geom_bad,
    "pairs_TRAIN": ntr, "pairs_DEV": ndv, "pairs_TEST_sealed": nte,
    "components_TRAIN": cnt.get('TRAIN',0), "components_DEV": cnt.get('DEV',0), "components_TEST": cnt.get('TEST',0),
    "rows_with_N": n_amb_rows, "N_bases": amb_bases, "homology": homology,
    "reference": "Ensembl GRCh37 primary_assembly (== hg19)", "cap_removed": True,
    "runtime_s": round(time.time()-T0, 1)}
json.dump(summary, open(os.path.join(OUT, 'cohort_summary.json'), 'w', encoding='utf-8'), indent=2)
json.dump({"salt_split": SALT_SPLIT, "bins": {"TRAIN": "mod100<70", "DEV": "70-84", "TEST": ">=85"},
           "counts": {"TRAIN": cnt.get('TRAIN',0), "DEV": cnt.get('DEV',0), "TEST": cnt.get('TEST',0)},
           "canonical_id": "min member sequenceSHA"},
          open(os.path.join(OUT, 'split_manifest.json'), 'w', encoding='utf-8'), indent=2)
log('DONE')
