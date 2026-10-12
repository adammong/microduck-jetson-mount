"""Measure exact plans on held-out text, without moving the robot."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapter',type=Path);ap.add_argument('--unguarded',action='store_true');ap.add_argument('--cases',type=Path,default=ROOT/'results/language-v07/holdout.json');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cases=json.loads(a.cases.read_text())
    trained={json.loads(line)['messages'][1]['content'].lower() for split in ['train','valid'] for line in (ROOT/f'results/language-v07/data/{split}.jsonl').read_text().splitlines()}
    assert all(c['text'].lower() not in trained for c in cases),'Exact holdout overlap'
    cmd=[str(ROOT/'tmp/language-venv/bin/python'),str(ROOT/'training/language_command.py'),'--model',str(ROOT/'tmp/language-model')]
    if a.adapter:cmd+=['--adapter',str(a.adapter)]
    if a.unguarded:cmd+=['--unguarded']
    env=os.environ.copy();env['HF_HUB_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
    result=subprocess.run(cmd,input=''.join(json.dumps(dict(text=c['text']))+'\n' for c in cases),capture_output=True,text=True,check=True,timeout=120,env=env)
    answers=[json.loads(line) for line in result.stdout.splitlines()];assert len(answers)==len(cases)
    records=[]
    for case,answer in zip(cases,answers):
        exact=(answer.get('accepted') and answer.get('steps')==case['steps']) if case['steps'] else not answer.get('accepted')
        records.append(dict(**case,response=answer,exact=bool(exact)))
    valid=[r for r in records if r['steps']];rejects=[r for r in records if not r['steps']]
    data=dict(total=len(records),exact=sum(r['exact'] for r in records),supported=dict(total=len(valid),exact=sum(r['exact'] for r in valid)),unsupported=dict(total=len(rejects),rejected=sum(r['exact'] for r in rejects)),records=records,exact_training_requests_disjoint=True,model= json.loads((ROOT/'tmp/language-model/download.json').read_text()),adapter_sha256=hashlib.sha256((a.adapter/'adapters.safetensors').read_bytes()).hexdigest() if a.adapter else None)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k in ['total','exact','supported','unsupported']}))
if __name__=='__main__':main()
