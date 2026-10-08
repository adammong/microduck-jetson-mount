"""Package the current CAD revision and pinned rebuild inputs, excluding caches."""
from pathlib import Path
import json,zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    revision=json.loads((ROOT/'easy_mount/parameters.json').read_text())['revision']
    parts=['saddle','front_gate','jetson_tray','jetson_gate_left','jetson_gate_right']
    assert sorted(p.stem for p in (ROOT/'easy_mount/STL').glob('*.stl'))==sorted(parts)
    files=set()
    for folder in ('easy_mount/src','easy_mount/STEP','easy_mount/STL','easy_mount/simulation',
                   'easy_mount/checks','simulation/reference','STEP/imported','checks','training','docs',
                   f'results/mac-{revision}','results/mac-v05'):
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(p for p in (ROOT/'easy_mount').glob('*') if p.is_file())
    files.update(ROOT/p for p in ('README.md','THIRD_PARTY.md','requirements.txt','GLB/jetson_devkit.glb'))
    files.update(ROOT/'easy_mount/GLB'/f'{name}_{revision}.glb'
                 for name in ('microduck_easy_mount','equipped_mount','head_wiring_inspection'))
    dest=ROOT/'dist'/f'microduck-jetson-mount-{revision}.zip';dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as packet:
        for p in sorted(files):packet.write(p,p.relative_to(ROOT))
    print(dest)

if __name__=='__main__':main()
