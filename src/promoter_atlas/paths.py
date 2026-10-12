"""Repository assets for the supported editable-checkout installation."""
import os
from pathlib import Path
ROOT = Path(os.environ.get("PROMOTER_ATLAS_HOME", Path(__file__).resolve().parents[2])).resolve()
MANIFESTS = ROOT / "artifacts" / "manifests"
MODELS = ROOT / "models"
