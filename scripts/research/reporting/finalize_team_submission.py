import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path('E:/promoter_atlas')
BASE = ROOT / 'deliverables/competition_joint_20261004'
SRC = BASE / '官方代码复核 中文路径'
STAGE = BASE / '完整提交_ors5GHPs'
DELIVERY = BASE / 'delivery'
RUN = DELIVERY / '完整包 外部运行核验'
EXTRACT = DELIVERY / '完整包 解压核验'
TEAM = 'AI驱动的启动子核酸元件模块化设计与Bxb1定点整合系统验证平台'
TRACK = '赛道二：AI基因编辑与核酸工具设计'
ZIP = ROOT / 'release' / ('ors5GHPs_' + TEAM + '_代码提交.zip')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

class StripDocs(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body.pop(0)
        return node

def structure(p):
    return ast.dump(StripDocs().visit(ast.parse(p.read_text(encoding='utf-8-sig'))), include_attributes=False)

def rows(p, delim=','):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delim))

def compare(a, b, key, cols, delim=','):
    aa = {r[key]: r for r in rows(a, delim)}
    bb = {r[key]: r for r in rows(b, delim)}
    assert aa.keys() == bb.keys()
    diff = max(abs(float(aa[k][col]) - float(bb[k][col])) for k in aa for col in cols)
    assert math.isfinite(diff) and diff <= 1e-5, diff
    return {'rows': len(aa), 'max_absolute_difference': diff, 'tolerance': 1e-5}

pyfiles = list(STAGE.glob('*.py'))
assert all(structure(p) == structure(SRC/p.name) for p in pyfiles)
weights = {p.name: sha(p) for p in STAGE.glob('weight_shard_*.pt')}
assert len(weights) == 3
assert all(d == sha(SRC/n) for n, d in weights.items())
precheck = compare(RUN/'results.csv', SRC/'results.csv', 'candidate_id', ['Score_E','Score_S','Score_B'])
shutil.copyfile(RUN/'results.csv', STAGE/'results.csv')
checks = {'team_id': 'ors5GHPs', 'team_name': TEAM, 'track': TRACK,
          'python_ast_equivalence': {'passed': True, 'files': len(pyfiles), 'excluded': 'docstrings'},
          'weights_sha256': weights, 'esb_external_working_directory_cpu_check': precheck,
          'package_layout': 'all files at ZIP root', 'runtime': 'CPU, 1 thread',
          'weight_selection': 'epoch1 selected by the recorded DEV rule'}
(STAGE/'SUBMISSION_CHECKS.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
index = STAGE/'PROJECT_STRUCTURE.md'
index.write_text(index.read_text(encoding='utf-8').replace('PACKAGE_CHECKS.json / SHA256SUMS.txt', 'PACKAGE_CHECKS.json / SUBMISSION_CHECKS.json / SHA256SUMS.txt'), encoding='utf-8')
files = sorted(p for p in STAGE.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
(STAGE/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='utf-8')
files = sorted(p for p in STAGE.iterdir() if p.is_file())
ZIP.parent.mkdir(exist_ok=True)
assert not ZIP.exists(), 'Existing release must be preserved'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, p.name)
assert not EXTRACT.exists(), 'Use a fresh extraction directory'
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None
    assert all('/' not in n and '\\' not in n for n in z.namelist())
    assert len(z.namelist()) == len(files)
    z.extractall(EXTRACT)
for line in (EXTRACT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    digest, name = line.split('  ', 1)
    assert sha(EXTRACT/name) == digest, name
env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
runtime_out = DELIVERY/'完整包 最终运行结果'
runtime_out.mkdir(exist_ok=True)
def run(name, args):
    cp = subprocess.run([sys.executable, '-B'] + list(map(str,args)), cwd=os.environ['TEMP'], env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
    (runtime_out/(name+'.log')).write_text(cp.stdout+'\n'+cp.stderr, encoding='utf-8')
    assert cp.returncode == 0, cp.stderr[-2000:]

run('main', [EXTRACT/'main.py', '--input', 'demo_input.tsv', '--outdir', runtime_out, '--device','cpu','--threads','1','--batch','1','--track',TRACK])
esb = compare(runtime_out/'results.csv', EXTRACT/'results.csv', 'candidate_id', ['Score_E','Score_S','Score_B'])
run('activity', [EXTRACT/'diagnostic_activity.py','--input',EXTRACT/'activity_demo.tsv','--out',runtime_out/'activity.tsv','--device','cpu'])
activity = compare(runtime_out/'activity.tsv',EXTRACT/'activity_results.tsv','seq_id',['activity_z','activity_log2rpm_pred'],'\t')
mixed = runtime_out/'mixed.tsv'
with mixed.open('w',encoding='utf-8',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['seq_id','sequence'])
    for n in [100,271,600]: w.writerow([f'length_{n}',('ACGT'*150)[:n]])
run('mixed', [EXTRACT/'diagnostic_activity.py','--input',mixed,'--out',runtime_out/'mixed_results.tsv','--device','cpu'])
mixed_rows=rows(runtime_out/'mixed_results.tsv','\t')
assert [int(r['length']) for r in mixed_rows]==[100,271,600]
assert all(math.isfinite(float(r['activity_log2rpm_pred'])) for r in mixed_rows)
report = {'zip':str(ZIP),'zip_sha256':sha(ZIP),'bytes':ZIP.stat().st_size,'root_files':len(files),'nested_paths':0,'zip_crc_check':'PASS','all_file_hashes':'PASS','fresh_extraction_main':esb,'fresh_extraction_activity':activity,'mixed_lengths':[100,271,600],'python_logic_unchanged':True,'team_id':'ors5GHPs','team_name':TEAM}
(DELIVERY/'完整提交包_最终核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
