from pathlib import Path
import sys,json,subprocess,hashlib
sys.path.insert(0,'training')
import numpy as np
from evaluate_motion import trial,session
from evaluate_transitions import SlewedPolicy
root=Path('results/terrain-v07');policy=session(root/'policies/selected/policy.onnx')
# One continuous run on a fresh map; the whole trial and contacts are retained.
schedule=[(2,[0,0,0]),(9,[.2,0,0]),(12,[.15,0,.3]),(15,[0,0,0])]
r=trial(SlewedPolicy(policy),[0,0,0],710,15,render=root/'terrain-demo-captioned',schedule=schedule,revision='v07',terrain='mixed_5mm_3deg',terrain_seed=1700,trace=True)
r['command_slew']=True;(root/'terrain-demo.json').write_text(json.dumps(r,indent=2)+'\n');print('Demo survived',r['survived'],'contacts',r['payload_selfcontact_fraction'],r['nonfoot_ground_fraction'],flush=True)
assert r['survived'] and r['payload_selfcontact_fraction']==0 and r['nonfoot_ground_fraction']==0
import imageio.v2 as imageio,imageio_ffmpeg
from PIL import Image
src=root/'terrain-demo-captioned.mp4';out=root/'terrain-demo.mp4'
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(src),'-vf','crop=iw:ih-64:0:64','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-an',str(out)],check=True)
reader=imageio.get_reader(out);meta=reader.get_meta_data();frames=[];count=0
for i,frame in enumerate(reader):
 assert frame.shape[:2]==(296,640)
 if i%5==0:frames.append(Image.fromarray(frame).resize((480,222)))
 count+=1
reader.close();assert count==375 and meta['fps']==25 and abs(meta['duration']-15)<.05
frames[0].save('docs/v07-terrain.gif',save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False)
(root/'media.json').write_text(json.dumps(dict(path=str(out),duration_s=15,frames=count,fps=25,crop_top_pixels=64,continuous=True,physics_seed=710,terrain_seed=1700,terrain='mixed_5mm_3deg',sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2)+'\n')
