from pathlib import Path
import sys,json,subprocess,hashlib
sys.path.insert(0,'training')
from evaluate_robust import measure
from evaluate_motion import session
import imageio.v2 as imageio,imageio_ffmpeg
from PIL import Image
root=Path('results/robust-v07');policy=session(root/'policies/candidate/policy.onnx')
case=dict(name='mixed_disturbance_demo',suite='trip_slip',terrain='bumps_5mm',spawn=None,command=[.2,0,0],events=[dict(kind='push',start=4.,duration=.12,force_body=[0,4,0]),dict(kind='snag',start=8.,duration=.08,window=1.,foot='right',force_body=[-2,0,0]),dict(kind='slip',start=11.,duration=.5,friction=.35)])
r=measure(policy,case,600,1600,render=root/'recovery-demo-captioned',seconds=15,keep_trace=True)
(root/'recovery-demo.json').write_text(json.dumps(r,indent=2)+'\n')
print('Fresh continuous demo',r['safe'],r['recoveries'],flush=True)
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(root/'recovery-demo-captioned.mp4'),'-vf','crop=iw:ih-64:0:64','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-an',str(root/'recovery-demo.mp4')],check=True)
reader=imageio.get_reader(root/'recovery-demo.mp4');frames=[];count=0
for i,frame in enumerate(reader):
 assert frame.shape[:2]==(296,640)
 if i%5==0:frames.append(Image.fromarray(frame).resize((480,222)))
 count+=1
reader.close()
frames[0].save('docs/v07-recovery.gif',save_all=True,append_images=frames[1:],duration=200,loop=0,optimize=False)
(root/'media.json').write_text(json.dumps(dict(path=str(root/'recovery-demo.mp4'),policy_kind='rejected_candidate',policy_sha256=hashlib.sha256((root/'policies/candidate/policy.onnx').read_bytes()).hexdigest(),duration_s=count/25,frames=count,fps=25,crop_top_pixels=64,continuous=True,physics_seed=600,terrain_seed=1600,safe=r['safe'],recovery_complete=r['recovery_complete'],sha256=hashlib.sha256((root/'recovery-demo.mp4').read_bytes()).hexdigest()),indent=2)+'\n')
