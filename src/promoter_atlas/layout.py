"""Resolve original dataset names in the organized repository or an older data tree."""
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
from .paths import ROOT

@lru_cache(maxsize=1)
def path_map():
    return json.loads((ROOT/'artifacts/manifests/PATH_MAP.json').read_text('utf-8'))['paths']

def canonical_relative(name):
    name=str(name).replace('\\','/')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name:
        raise ValueError('Expected a project-relative path: '+name)
    name=p.as_posix().lstrip('./')
    mapping=path_map()
    if name in mapping:return mapping[name]
    for i in range(len(p.parts)-1,0,-1):
        prefix='/'.join(p.parts[:i])
        if prefix in mapping:return mapping[prefix]+'/'+ '/'.join(p.parts[i:])
    return name

def canonical_path(root,name):
    base=Path(root).resolve();target=(base/canonical_relative(name)).resolve()
    if not target.is_relative_to(base):raise ValueError('Destination leaves project root')
    return target

def project_path(root,*parts):
    """Read either a pre-migration data tree or the current categorized tree."""
    rel='/'.join(str(x).replace('\\','/') for x in parts)
    canonical=canonical_path(root,rel)
    legacy=Path(root)/rel
    return str(canonical if canonical.exists() or not legacy.exists() else legacy)
