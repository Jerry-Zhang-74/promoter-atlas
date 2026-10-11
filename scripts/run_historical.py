"""Prepare an original-layout working tree and run a preserved historical Python script."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES={'.py','.ps1','.sh','.bat'}

def prepare(root,workspace):
    root=root.resolve();workspace=workspace.resolve()
    if workspace==root or root.is_relative_to(workspace):raise ValueError('Choose a child or separate workspace')
    marker=workspace/'.historical-workspace.json'
    if workspace.exists() and any(workspace.iterdir()) and not marker.exists():
        raise FileExistsError('Choose an empty workspace or an existing historical workspace')
    workspace.mkdir(parents=True,exist_ok=True)
    meta={'schema':1,'source_root':str(root),'workspace':str(workspace)}
    if marker.exists() and json.loads(marker.read_text('utf-8')).get('source_root')!=str(root):
        raise ValueError('Workspace belongs to another source checkout')
    marker.write_text(json.dumps(meta,indent=2),'utf-8')
    local=root/'.local/maintenance/layout-20261011/local-path-map.json'
    manifest=local if local.exists() else root/'artifacts/manifests/PATH_MAP.json'
    mapping=json.loads(manifest.read_text('utf-8'))['paths']
    counts={'copied_scripts':0,'linked_files':0,'reused':0,'unavailable':0}
    for old,new in mapping.items():
        # Runtime generation concerns research inputs and code; publishing caches and secrets stay separate.
        if old.split('/')[0] not in {'code','runs','models','data','features','labels','promoters','legacy_data','reports','figures','design','interpretation','reference','wetlab'}:continue
        src=(root/new).resolve();dst=(workspace/old).resolve()
        if not src.is_relative_to(root) or not dst.is_relative_to(workspace):raise ValueError('Unsafe mapping')
        if not src.is_file():
            if not src.is_dir():counts['unavailable']+=1
            continue
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():counts['reused']+=1;continue
        if src.suffix.lower() in SOURCE_SUFFIXES:
            text=src.read_text('utf-8-sig')
            target=workspace.as_posix()
            for prefix in ['E:/promoter_atlas','E:\\\\promoter_atlas','E:\\promoter_atlas','/mnt/e/promoter_atlas']:
                text=text.replace(prefix,target)
            dst.write_text(text,'utf-8');counts['copied_scripts']+=1
        else:
            # Independent work copies keep mutable logs, tables and manifests separate from the archive.
            # Large frozen inputs use links and are intended to be read by historical workflows.
            if src.stat().st_size<16*1024*1024 or src.suffix.lower() in {'.json','.md','.txt','.log'}:
                shutil.copy2(src,dst)
            else:
                try:os.link(src,dst)
                except OSError:
                    try:dst.symlink_to(src)
                    except OSError:raise OSError('Place the workspace on the same disk to link large inputs: '+str(src))
            counts['linked_files']+=1
    return counts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=str(HERE))
    ap.add_argument('--workspace',default=str(HERE/'workspaces/historical'))
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('script',nargs='?')
    ap.add_argument('arguments',nargs=argparse.REMAINDER)
    a=ap.parse_args();workspace=Path(a.workspace).resolve()
    print(json.dumps(prepare(Path(a.root),workspace),ensure_ascii=False),flush=True)
    if a.prepare_only:return
    if not a.script:ap.error('Provide the original project-relative script path')
    script=(workspace/a.script).resolve()
    if not script.is_relative_to(workspace) or not script.is_file():raise FileNotFoundError(a.script)
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    env=dict(os.environ);env['PYTHONPATH']=str(workspace)+os.pathsep+str(workspace/'runs/scripts')+os.pathsep+env.get('PYTHONPATH','')
    raise SystemExit(subprocess.call([sys.executable,str(script),*args],cwd=workspace,env=env))

if __name__=='__main__':main()
