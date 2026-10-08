"""Matched stock/loaded CPU PPO pilot (same seeds, budget and BAM physics)."""
from pathlib import Path
import argparse,os,subprocess,json,time,shutil
from paths import ROOT,model_root,revision
PYTHON=ROOT/'tmp/microduck-lab/microduck_local/.venv/bin/python'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=262144)
    ap.add_argument('--envs',type=int,default=8);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--variant',choices=('stock','loaded','both'),default='both')
    ap.add_argument('--allow-nonfoot-support',action='store_true',help='Reproduce the original pilot; permits the backpack to support a collapsed robot')
    ap.add_argument('--init-from',type=Path,help='Resume a native run directory; select one variant')
    ap.add_argument('--run-tag',default='',help='Suffix for a new output directory')
    ap.add_argument('--lr',type=float,default=None,help='Override the PPO learning rate')
    args=ap.parse_args()
    if args.init_from and args.variant=='both':ap.error('--init-from needs a single --variant')
    for variant in ('stock','loaded') if args.variant=='both' else [args.variant]:
        name=f'{revision()}-{variant}-bam-s{args.seed}-{args.steps}'+('' if args.allow_nonfoot_support else '-strict')+('-warm' if args.init_from else '')+('-'+args.run_tag if args.run_tag else '');out=ROOT/'tmp/mac-runs'/name
        if out.exists():raise SystemExit(f'Run exists: {out}. Choose --run-tag retry to preserve it.')
        out.mkdir(parents=True)
        env=os.environ.copy();env.update(MICRODUCK_RL_DIR=str(model_root()/variant),
            MICRODUCK_RUNS_DIR=str(ROOT/'tmp/mac-runs'),MICRODUCK_ACTUATOR='bam',
            MICRODUCK_RUN_TITLE=f'Payload study: {variant}, seed {args.seed}',OMP_NUM_THREADS='1',MOUNT_STRICT_FLOOR='0' if args.allow_nonfoot_support else '1')
        cmd=[str(PYTHON),str(ROOT/'training/train_policy.py'),'--device','cpu','--actuator','bam',
            '--steps',str(args.steps),'--envs',str(args.envs),'--seed',str(args.seed),
            '--run-name',name,'--snap-steps','65536','--no-domain-rand','--no-obs-noise']
        if args.init_from:
            donor=args.init_from.resolve()
            if not (donor/'model.zip').exists() and (donor/'model.sb3').exists():
                restored=ROOT/'tmp/resume'/name;restored.mkdir(parents=True,exist_ok=True)
                shutil.copy2(donor/'model.sb3',restored/'model.zip')
                for filename in ('vecnormalize.pkl','run.json'):shutil.copy2(donor/filename,restored/filename)
                donor=restored
            cmd.extend(['--init-from',str(donor)])
            record=json.loads((donor/'run.json').read_text())
            if record.get('keep_command_normalizer'):
                cmd.append('--freeze-obs-norm')
                env['MOUNT_KEEP_COMMAND_NORMALIZER']='1'
        if args.lr is not None:cmd.extend(['--lr',str(args.lr)])
        started=time.time();print('Training',name,flush=True)
        with (out/'console.log').open('w') as log:subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(PYTHON),'-m','microduck_local.export_onnx',str(out)],env=env,cwd=ROOT,check=True)
        if args.init_from and record.get('keep_command_normalizer'):
            saved=json.loads((out/'run.json').read_text())
            saved['keep_command_normalizer']=True
            saved['teacher_import_source']=str(args.init_from)
            (out/'run.json').write_text(json.dumps(saved,indent=2)+'\n')
        meta=dict(variant=variant,seed=args.seed,requested_steps=args.steps,envs=args.envs,
            elapsed_s=time.time()-started,mount_revision=revision(),actuator='bam',device='cpu',domain_randomization=False,
            requested_learning_rate=args.lr,init_from=str(args.init_from) if args.init_from else None,
            observation_noise=False,contact_model='groundcontact',assist=False,symmetry_loss=False,strict_nonfoot_floor=not args.allow_nonfoot_support,
            harness_commit='bbf0326ef97975f7062368914e0729504d17226f')
        (out/'study.json').write_text(json.dumps(meta,indent=2)+'\n');print('Finished',meta,flush=True)
if __name__=='__main__':main()
