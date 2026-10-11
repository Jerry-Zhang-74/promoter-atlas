"""Assemble the flat Njoint competition bundle in a staging directory."""
from pathlib import Path
import ast, csv, hashlib, io, json, shutil, sys, tokenize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'release/staging_batch52_njoint_20261004'
CORE=ROOT/'runs/breakthrough_20261002/code/batch53_model_core'
REVIEWED=ROOT/'releases/batch52_second_model/v6.38.0-njoint-reviewed'
COHORT=ROOT/'data/breakthrough_20261002/batch52/corrected/cohort_v2'
OUT.mkdir(parents=True,exist_ok=True)
VERSION='v6.38.0-batch52-njoint-competition-20261004'
AUDIT=[]


def put(name,body):
    (OUT/name).write_text(body.strip()+'\n',encoding='utf-8')


def jput(name,obj):
    put(name,json.dumps(obj,ensure_ascii=False,indent=2))


class StripDocs(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        return self.generic_visit(node)


def semantic(source):
    return ast.dump(StripDocs().visit(ast.parse(source)),include_attributes=False)


def clean_source(path):
    text=path.read_text(encoding='utf-8-sig')
    lines=text.splitlines(keepends=True);offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    edits=[]
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                a=offsets[first.lineno-1]+first.col_offset
                b=offsets[first.end_lineno-1]+first.end_col_offset
                edits.append((a,b,'""'))
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type==tokenize.COMMENT:
            edits.append((offsets[t.start[0]-1]+t.start[1],offsets[t.end[0]-1]+t.end[1],''))
    for a,b,s in sorted(edits,reverse=True):text=text[:a]+s+text[b:]
    text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
    while '\n\n\n' in text:text=text.replace('\n\n\n','\n\n')
    assert semantic(text)==semantic(path.read_text(encoding='utf-8-sig')),path
    AUDIT.append({'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'executable_ast_preserved':True})
    return text


def prepare():
    names=['model_core.py','spec.py','normalizer.py','checkpoint.py','io_utils.py','losses.py','metrics.py','train_neural.py','data_adapter.py','epoch_cap.py','exec_gate.py','_exec_gate_vendored.py','r0_baseline.py']
    for name in names:put(name,clean_source(CORE/name))
    vendor=hashlib.sha256((OUT/'_exec_gate_vendored.py').read_bytes()).hexdigest()
    gate=(OUT/'exec_gate.py').read_text()
    gate=gate.replace('e1ff80a6e411d0989f7bfce3bd76950dabf8afb4f876e6b3c7e70692fc272571',vendor)
    put('exec_gate.py',gate)
    cohort=clean_source(ROOT/'code/batch52_data_v2/cohort_v2.py')
    start=cohort.index('HELPER = ');end=cohort.index('from data_adapter',start)
    cohort=cohort[:start]+cohort[end:]
    put('cohort_v2.py',cohort)
    # Dataset construction and evaluation sources retain their original relative mapping.
    sources={}
    for name in ['build_cohort_v2.py','eval_v2.py','verify_v2.py','predict_v2.py','run_all_v2.py']:
        p=ROOT/'code/batch52_data_v2'/name
        dest='source__code__batch52_data_v2__'+name
        put(dest,clean_source(p));sources[dest]=str(p.relative_to(ROOT)).replace('\\','/')
    for name in ['pipeline_coordinator.py','eval_replay.py','gpu_lock.py']:
        p=CORE/name;dest='source__runs__breakthrough_20261002__code__batch53_model_core__'+name
        put(dest,clean_source(p));sources[dest]=str(p.relative_to(ROOT)).replace('\\','/')
    for name in names:sources[name]='runs/breakthrough_20261002/code/batch53_model_core/'+name
    sources['cohort_v2.py']='code/batch52_data_v2/cohort_v2.py'
    jput('source_manifest.json',sources)
    ck='Njoint_selected.ckpt'
    shutil.copy2(REVIEWED/'model'/ck,OUT/ck)
    shutil.copy2(REVIEWED/'model'/(ck+'.sha256.json'),OUT/(ck+'.sha256.json'))
    side=json.loads((OUT/(ck+'.sha256.json')).read_text());side['path']=ck
    jput(ck+'.sha256.json',side)
    shutil.copy2(REVIEWED/'config/config.json',OUT/'config.json')
    shutil.copy2(REVIEWED/'normalizer/normalizer.json',OUT/'normalizer.json')
    man=json.loads((REVIEWED/'MODEL_MANIFEST.json').read_text())
    man.pop('staging',None);man.pop('evidence_scope',None)
    man['format']='competition.batch52.v1';man['version']=VERSION;man['model_id']='batch52_Njoint'
    man['checkpoint']['path']=ck;man['units']='Log2RPM_HEK293'
    man['data_context']={'cell_line':'HEK293','input_length_bp':[100,600],'reference_assembly':'GRCh37 / hg19','labels':'GSE301246 v2_MH4 fold0','evaluation':'retrospective frozen group split','local_platform':'Bxb1 platform / natural promoter data available','candidate_assay_status':'planned'}
    man['checkpoint_field_notes']={'global_step':'number of training rows processed in the recorded epoch'}
    jput('MODEL_MANIFEST.json',man)
    shutil.copy2(ROOT/'reports/breakthrough_20261002/batch52/final_test_metrics.json',OUT/'final_test_metrics.json')
    shutil.copy2(COHORT/'groups.tsv',OUT/'group_splits.tsv')
    cm=json.loads((COHORT/'manifest.json').read_text());cm['source_bed']=Path(cm['source_bed'].replace('\\','/')).name
    cm['reference_assembly']='GRCh37 / hg19'
    cm['reference_sha256']='0a43b56dec40debae976d6e70cac68ea6ed874f9fb7c8c814363702ff1d47865'
    jput('data_manifest.json',cm)
    for arm in ['Nabs','Njoint']:
        ready=json.loads((COHORT.parent/'models'/(arm+'_ready.json')).read_text())
        hist=[]
        for r in ready['history']:
            hist.append({'epoch':r['epoch']+1,'loss':r['loss'],'dev_rho':r['dev']['rho'],'dev_mae':r['dev']['mae'],'dev_rmse':r['dev']['rmse'],'dev_pair_mae':r['dev_pair_mae'],'dev_sign':r['dev_sign']})
        jput(arm+'_training_history.json',hist)
    model_report=ROOT/'outputs/batch52模型选择与更新说明_20261004.txt'
    shutil.copy2(model_report,OUT/'模型选择说明.txt')
    jput('submission_metadata.json',{'version':VERSION,'date':'2026-10-04','track':'赛道二：AI基因编辑与核酸工具设计','team':'AI驱动的启动子核酸元件模块化设计与Bxb1定点整合系统验证平台(ors5GHPs)','model':'batch52 Njoint','layout':'all files at ZIP root'})
    put('requirements.txt','numpy==2.5.3\ntorch==2.11.0')
    jput('source_edit_audit.json',{'edits':'comments and docstrings simplified; flat imports and vendored source checksum adapted','checks':AUDIT})


def samples():
    selected=[];groups=set()
    with (COHORT/'endpoints.tsv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if r['split']=='TEST' and r['union_id'] not in groups:
                groups.add(r['union_id']);selected.append(r)
                if len(selected)==12:break
    targets={r['coord_id'] for r in selected};seqs={};sid=None;buf=[]
    def flush():
        if sid in targets:seqs[sid]=''.join(buf)
    with (COHORT/'sequences.fa').open(encoding='utf-8') as f:
        for line in f:
            if line.startswith('>'):
                flush()
                if len(seqs)==len(targets):break
                sid=line[1:].strip();buf=[]
            else:buf.append(line.strip())
        flush()
    assert len(seqs)==12
    with (ROOT/'reports/breakthrough_20261002/batch54/Njoint_TEST_parent_granted.tsv').open() as f:
        pred={r['coord_id']:r['activity_pred'] for r in csv.DictReader(f,delimiter='\t') if r['coord_id'] in targets}
    put('demo_input.fasta','\n'.join(f'>DEMO{i:02d}\n{seqs[r["coord_id"]]}' for i,r in enumerate(selected,1)))
    with (OUT/'demo_provenance.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['seq_id','coord_id','union_id','split','length','measured_Log2RPM_HEK293','source'])
        for i,r in enumerate(selected,1):w.writerow([f'DEMO{i:02d}',r['coord_id'],r['union_id'],r['split'],r['length'],r['Log2RPM_HEK293'],'GSE301246 v2_MH4 fold0'])
    with (OUT/'demo_expected.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['seq_id','length','activity_pred_Log2RPM'])
        for i,r in enumerate(selected,1):w.writerow([f'DEMO{i:02d}',r['length'],pred[r['coord_id']]])


if __name__=='__main__':
    prepare();samples();print(OUT)
