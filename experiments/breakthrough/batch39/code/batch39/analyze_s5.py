import json, re, collections, hashlib
s5=json.load(open('data/breakthrough_20261002/batch39/s5_rows_raw.json',encoding='utf-8'))
pat=re.compile(r'^(?P<name>.+?)::(?P<chr>chr[0-9XYM]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-])$')
rows=[]; n_real=0; n_scram=0; orient=collections.Counter(); chroms=collections.Counter()
res={}
for r in s5:
    m=pat.match(r['A'])
    ctrl=int(r['F'])
    rec=dict(label=r['A'],ctrl=ctrl,stat=r['E'],z=r['G'],mad=r['H'],
             p_emp=r['I'],p_mad=r['J'],p_z=r['K'])
    if m:
        n_real+=1
        rec.update(parsed=True,name=m.group('name'),chrom=m.group('chr'),
                   start=int(m.group('start')),end=int(m.group('end')),strand=m.group('strand'))
        orient[m.group('strand')]+=1; chroms[m.group('chr')]+=1
        rec['span']=rec['end']-rec['start']
    else:
        n_scram+=1
        rec.update(parsed=False,name=r['A'],chrom=None,start=None,end=None,strand=None,span=None)
    rows.append(rec)

# consistency: parsed coords vs columns B/C/D
mism=0
for r,src in zip(rows,s5):
    if r['parsed'] and (src['B']!=r['chrom'] or src['C']!=str(r['start']) or src['D']!=str(r['end'])): mism+=1

# families: prefix = name with trailing _<digits> stripped
def fam(n): return re.sub(r'_\d+$','',n)
famc=collections.Counter(fam(r['name']) for r in rows if r['parsed'])
scram_names=collections.Counter(r['name'] for r in rows if not r['parsed'])
spans=collections.Counter(r['span'] for r in rows if r['parsed'])

summary={
 'n_data_rows':len(s5),'n_real':n_real,'n_scrambled':n_scram,
 'ctrl_col_0':sum(1 for r in rows if r['ctrl']==0),'ctrl_col_1':sum(1 for r in rows if r['ctrl']==1),
 'scrambled_label_vs_ctrlcol_agree': all((not r['parsed'])==bool(r['ctrl']) for r in rows),
 'orientation':dict(orient),'chroms':dict(chroms),
 'coord_col_mismatch':mism,
 'distinct_coords':len({(r['chrom'],r['start'],r['end']) for r in rows if r['parsed']}),
 'span_values':dict(spans),
 'n_families':len(famc),'top_families':famc.most_common(15),
 'scram_name_examples':[k for k,_ in scram_names.most_common(3)],
 'dup_labels': len(s5)-len({r['A'] for r in s5}),
}
json.dump({'summary':summary}, open('data/breakthrough_20261002/batch39/s5_audit.json','w',encoding='utf-8'), indent=1)
for k,v in summary.items(): print(k,'=',v)

# ---- label-free grouping: connected components of overlapping intervals per chromosome ----
bychr=collections.defaultdict(list)
for r in rows:
    if r['parsed']: bychr[r['chrom']].append((r['start'],r['end'],r['name'],r['strand'],r['label']))
groups=[]
for c,iv in bychr.items():
    iv=sorted(iv)
    cur=None
    for s,e,name,st,lab in iv:
        if cur and s<=cur[1]:
            cur[1]=max(cur[1],e); cur[2].append(lab)
        else:
            if cur: groups.append(cur)
            cur=[c,s,e,[lab]]
    if cur: groups.append(cur)
sizes=collections.Counter(len(g[3]) for g in groups)
json.dump({'n_groups':len(groups),'group_size_dist':dict(sorted(sizes.items())),
           'largest_groups':sorted(((len(g[3]),g[0],g[1],g[2]) for g in groups),reverse=True)[:10]},
          open('data/breakthrough_20261002/batch39/grouping.json','w',encoding='utf-8'),indent=1)
print('n_groups',len(groups),'size_dist',dict(sorted(sizes.items())))
