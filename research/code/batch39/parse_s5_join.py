import zipfile, re, json, collections, sys
from xml.etree import ElementTree as ET
wb='data/breakthrough_20261002/batch36/mapt/media-1.xlsx'
import hashlib
print('workbook_sha256', hashlib.sha256(open(wb,'rb').read()).hexdigest())
z=zipfile.ZipFile(wb)
# shared strings
ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
sst=[]
try:
    r=ET.fromstring(z.read('xl/sharedStrings.xml'))
    for si in r:
        sst.append(''.join(t.text or '' for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')))
except KeyError:
    pass
print('shared_strings',len(sst))
def read_sheet(name):
    root=ET.fromstring(z.read(name))
    rows=[]
    for row in root.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row'):
        cells={}
        for c in row:
            ref=c.get('r'); col=re.match(r'([A-Z]+)',ref).group(1)
            t=c.get('t'); v=c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
            if v is None: 
                isv=c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}is')
                val=''.join(x.text or '' for x in isv.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t')) if isv is not None else ''
            else:
                val=v.text or ''
                if t=='s': val=sst[int(val)]
            cells[col]=val
        rows.append(cells)
    return rows
# find s5 sheet part
wbxml=ET.fromstring(z.read('xl/workbook.xml'))
sheets=[(s.get('name'), s.get('sheetId')) for s in wbxml.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet')]
print('sheets',sheets)
rows=read_sheet('xl/worksheets/sheet6.xml')
print('s5 rows incl header',len(rows))
hdr=rows[0]
print('header',[hdr.get(c) for c in ['A','B','C','D','E','F','G','H','I','J','K']])
data=rows[1:]
print('data rows',len(data))
out=[]
for r in data:
    out.append({c:r.get(c,'') for c in ['A','B','C','D','E','F','G','H','I','J','K']})
json.dump(out, open('data/breakthrough_20261002/batch39/s5_rows_raw.json','w',encoding='utf-8'))
# counts
lab=[r['A'] for r in out]
print('first3 labels',lab[:3])
n_scram=sum(1 for l in lab if 'scram' in l.lower())
print('scram_like_labels',n_scram)
print('distinct labels',len(set(lab)))
# control column values
ctr=collections.Counter(r['F'] for r in out)
print('control_col_top',ctr.most_common(6))
# parse coord from label suffix ::chr:start-end
import re as _re
pat=_re.compile(r'::(chr[0-9XYM]+):(\d+)-(\d+)([+-])?$')
bad=0; coords=set(); chroms=collections.Counter()
for r in out:
    m=pat.search(r['A'])
    if not m: bad+=1; continue
    chroms[m.group(1)]+=1
    coords.add((m.group(1),int(m.group(2)),int(m.group(3))))
print('label_coord_parse_bad',bad,'distinct_coords',len(coords),'chroms',chroms.most_common())
# compare label coord vs columns C/D
mism=0
for r in out:
    m=pat.search(r['A'])
    if not m: continue
    if r['B']!=m.group(1) or r['C']!=m.group(2) or r['D']!=m.group(3): mism+=1
print('label_vs_colBCD_mismatch',mism)
json.dump({'n_rows':len(out),'n_scram_like':n_scram,'distinct_labels':len(set(lab)),'control_top':ctr.most_common(6),'coord_parse_bad':bad,'distinct_coords':len(coords),'label_vs_colBCD_mismatch':mism}, open('data/breakthrough_20261002/batch39/s5_summary.json','w',encoding='utf-8'),indent=1)
