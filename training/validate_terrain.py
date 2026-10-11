"""Choose correction strength on validation seeds before final terrain testing."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from terrain_policy import TerrainPolicy,BASE,export
from evaluate_motion import session
import evaluate_terrain

def score(args):
 scale,parameters=args;evaluate_terrain._actor=TerrainPolicy(session(BASE),np.asarray(parameters)*scale)
 rows=[evaluate_terrain.job((terrain,task,100+i,1000+i,12.,i==1,1.)) for terrain in ['flat','bumps_5mm','uphill_3deg','downhill_3deg','cross_left_3deg','mixed_5mm_3deg'] for task in ['forward','backward','left','turn_left'] for i in range(2)]
 return dict(scale=scale,parameters=(np.asarray(parameters)*scale).tolist(),passed=sum(r['passed'] for r in rows),flat_passed=sum(r['passed'] for r in rows if r['terrain']=='flat'),mean_scaled_error=float(np.mean([r['scaled_tracking_error'] for r in rows])),trials=rows)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--search',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policy-out',type=Path,required=True);a=ap.parse_args();parameters=json.loads(a.search.read_text())['best']['parameters']
 with mp.get_context('spawn').Pool(4) as pool:rows=pool.map(score,[(scale,parameters) for scale in [0.,.25,.5,.75,1.]])
 baseline=rows[0];eligible=[r for r in rows if r['flat_passed']>=baseline['flat_passed']]
 best=max(eligible,key=lambda r:10*r['passed']-r['mean_scaled_error']);export(best['parameters'],a.policy_out)
 a.out.write_text(json.dumps(dict(physics_seeds=[100,101],terrain_seeds=[1000,1001],randomized_seeds=[101],selection='Maximize 10*passes - mean scaled error; do not reduce flat validation passes',candidates=rows,selected_scale=best['scale'],selected_parameters=best['parameters']),indent=2)+'\n')
 print('Validation chose scale',best['scale'],'passes',best['passed'],'/',len(best['trials']),'baseline',baseline['passed'],flush=True)
