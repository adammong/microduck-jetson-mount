"""Build matched stock/loaded Mac scenes; CAD XML remains authoritative."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, hashlib, copy
import mujoco
ROOT=Path(__file__).resolve().parents[1]
from paths import model_root,result_root
OUT=model_root()

def build():
    records=[]
    for variant in ('stock','loaded'):
        for contacts in ('walk','groundcontact'):
            source=(ROOT/'simulation/reference'/f'robot_{contacts}.xml' if variant=='stock'
                    else ROOT/'easy_mount/simulation'/f'microduck_easy_mount_{contacts}.xml')
            robot=ET.parse(source).getroot()
            robot.find('compiler').attrib.pop('meshdir',None)
            for asset in robot.findall('.//asset/*'):
                if 'file' in asset.attrib:
                    asset.set('file',str((source.parent/('assets' if variant=='stock' else '')/asset.get('file')).resolve()))
            if variant=='loaded':
                payload=robot.find('.//body[@name="easy_mount_payload"]')
                board=copy.deepcopy(payload.find('geom[@name="easy_jetson"]'))
                board.attrib.update(name='jetson_envelope_visual',group='2',contype='0',conaffinity='0',rgba='0.15 0.25 0.18 1')
                payload.append(board)
            scene=ET.parse(ROOT/'tmp/sim-inputs/scene_walk.xml').getroot()
            scene.remove(scene.find('include'))
            for child in scene: robot.append(child)
            # Training and evaluation both use the fuller groundcontact model.
            # The walk model is retained only as a diagnostic of contact sensitivity.
            folder=OUT/variant/'src/mjlab_microduck/robot/microduck';folder.mkdir(parents=True,exist_ok=True)
            dest=folder/('scene_walk.xml' if contacts=='groundcontact' else 'scene_feet_only.xml')
            ET.indent(robot);ET.ElementTree(robot).write(dest,encoding='utf-8',xml_declaration=True)
            if contacts=='groundcontact': (folder/'scene.xml').write_bytes(dest.read_bytes())
            compiled=mujoco.MjModel.from_xml_path(str(dest))
            mujoco.mj_saveModel(compiled,str(dest.with_suffix('.mjb')),None)
            del compiled
            records.append(dict(variant=variant,contacts=contacts,source=str(source.relative_to(ROOT)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scene=str(dest.relative_to(ROOT))))
    (result_root()/'scenes.json').write_text(json.dumps(records,indent=2)+'\n')
    return OUT
if __name__=='__main__': print(build())
