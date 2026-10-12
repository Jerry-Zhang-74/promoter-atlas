"""Bootstrap helpers for the flat-ZIP CLI wrappers.

This module is intentionally small and dependency-free.  It locates the real
project tree (the directory that contains ``src/promoter_atlas/model.py``) and
restores it from the flat ZIP metadata map when necessary.

No network access, no credentials, no bytecode writing.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.dont_write_bytecode = True

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_REL = os.path.join("src", "promoter_atlas", "model.py")
_FILE_MAP = "FILE_MAP.json"
_RESTORE_SCRIPT = "restore_project.py"
_RUNTIME_DIR = "runtime_project"


def _has_project_tree(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _MODEL_REL))


def project_root() -> str:
    """Return the directory that contains the real project tree.

    Resolution order:

    1. If this wrapper directory itself contains ``src/promoter_atlas/model.py``
       it is the project root.
    2. Otherwise, if ``FILE_MAP.json`` exists next to this file, run
       ``restore_project.py --target runtime_project`` with the current Python
       interpreter and return ``<here>/runtime_project``.

    The restore script is responsible for not overwriting differing restored
    files.  This helper never modifies project sources.
    """
    if _has_project_tree(_HERE):
        return _HERE

    file_map = os.path.join(_HERE, _FILE_MAP)
    restore = os.path.join(_HERE, _RESTORE_SCRIPT)
    if os.path.isfile(file_map):
        if not os.path.isfile(restore):
            raise RuntimeError(
                "FILE_MAP.json found but restore_project.py is missing; "
                "cannot restore the project tree"
            )
        target = os.path.join(_HERE, _RUNTIME_DIR)
        cmd = [sys.executable, restore, "--target", _RUNTIME_DIR]
        proc = subprocess.run(cmd, cwd=_HERE)
        if proc.returncode != 0:
            raise RuntimeError(
                "restore_project.py failed with exit code %d" % proc.returncode
            )
        if not _has_project_tree(target):
            raise RuntimeError(
                "restore_project.py completed but %s was not produced" % _MODEL_REL
            )
        return target

    raise RuntimeError(
        "cannot locate project tree: no local src/promoter_atlas/model.py and "
        "no FILE_MAP.json next to %s" % _HERE
    )


def resolve_path(path: str, base: str) -> str:
    """Resolve *path* against *base* unless it is already absolute."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))
