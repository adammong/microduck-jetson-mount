"""Reproduce the paired push regression, without padding or hiding a fall."""
from pathlib import Path
import sys,json,subprocess,hashlib
sys.path.insert(0,'training')
from evaluate_robust import measure,cases
from evaluate_motion import session
import imageio.v2 as imageio,imageio_ffmpeg
root=Path('results/robust-v07');case=next(c for c in cases('push') if c['name']=='flat_forward_6N_y')
rows=[]
for label,path in [('baseline',root/'policies/selected/policy.onnx'),('candidate',root/'policies/candidate/policy.onnx')]:
 r=measure(session(path),case,100,1100,render=root/('push-'+label+'-captioned'),keep_trace=True)
 assert r['safe']==(label=='baseline')
 name=root/('push-'+label+'.mp4')
 subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-i',str(root/('push-'+label+'-captioned.mp4')),'-vf','crop=iw:ih-64:0:64','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-an',str(name)],check=True)
 reader=imageio.get_reader(name);frames=sum(1 for _ in reader);reader.close()
 r['media']=dict(path=str(name),frames=frames,fps=25,duration_s=frames/25,policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),crop_top_pixels=64)
 rows.append(dict(label=label,result=r))
 print(label,r['safe'],r['seconds'],frames,flush=True)
(root/'push-regression-demo.json').write_text(json.dumps(dict(description='Same held-out trial: baseline survives, full candidate falls after the second 6 N lateral shove.',simulation_only=True,trials=rows),indent=2)+'\n')
