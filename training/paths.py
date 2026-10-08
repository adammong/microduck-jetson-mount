"""Revision-aware artifact paths; old simulation results remain immutable."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def revision():return json.loads((ROOT/'easy_mount/parameters.json').read_text())['revision']
def model_root(rev=None):return ROOT/'tmp/mac-models'/(rev or revision())
def result_root(rev=None):
    path=ROOT/'results'/('mac-'+(rev or revision()));path.mkdir(parents=True,exist_ok=True);return path
