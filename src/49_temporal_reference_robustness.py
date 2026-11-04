#!/usr/bin/env python3
"""Temporal robustness on saved Script 42 version-level outputs, no reconstruction.

Tests distance to earliest observed version, immediate previous observed version,
and angular distance on the same cached E5 vectors. Also constructs a within-
family permutation null for the earliest-reference association. This does NOT
substitute for an independent embedding model (MiniLM/MPNet not computed).

python src/49_temporal_reference_robustness.py --permutations 199
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs'/'anxiety_lineages';OUT=BASE/'reviewer_experiments'/'temporal'
INPUT=BASE/'reviewer_experiments'/'downstream';EMBED=BASE/'e5_retrieval'/'e5_embeddings.npy'

def cos_dist(E,aa,bb):
 a=np.asarray(E[aa],dtype=np.float32);b=np.asarray(E[bb],dtype=np.float32)
 norm=np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
 return 1-np.clip(np.sum(a*b,axis=1)/norm,-1,1)

def fe(df,x,y):
 d=df[['lineage',x,y]].dropna().copy()
 d['xc']=d[x]-d.groupby('lineage')[x].transform('mean')
 d['yc']=d[y]-d.groupby('lineage')[y].transform('mean')
 d=d[d.groupby('lineage').xc.transform(lambda a:np.sum(a*a)>1e-12)]
 if len(d)<10 or d.lineage.nunique()<2:return {'beta':None,'n_versions':len(d),'n_families':int(d.lineage.nunique())}
 fit=sm.OLS(d.yc,d[['xc']]).fit(cov_type='cluster',cov_kwds={'groups':d.lineage})
 ci=fit.conf_int().loc['xc'].to_numpy()
 return {'beta':float(fit.params['xc']),'se':float(fit.bse['xc']),'ci_low':float(ci[0]),'ci_high':float(ci[1]),
         'n_versions':len(d),'n_families':int(d.lineage.nunique())}

def permute_null(df,reps,seed):
 d=df[['lineage','t_days','distance']].copy();d['x']=np.log1p(d.t_days.clip(lower=0))
 d=d.sort_values('lineage').reset_index(drop=True)
 d['x']=d.x-d.groupby('lineage').x.transform('mean')
 d['y']=d.distance-d.groupby('lineage').distance.transform('mean')
 d=d[d.groupby('lineage').x.transform(lambda v:float(np.dot(v,v))>1e-12)].copy()
 if d.empty:return None
 x=d.x.to_numpy();y=d.y.to_numpy();den=np.dot(x,x)
 observed=float(np.dot(x,y)/den)
 lengths=d.groupby('lineage',sort=False).size().to_numpy()
 cut=np.r_[0,np.cumsum(lengths)]
 rng=np.random.default_rng(seed);extreme=0
 for _ in range(reps):
  xp=x.copy()
  for lo,hi in zip(cut[:-1],cut[1:]):rng.shuffle(xp[lo:hi])
  beta=float(np.dot(xp,y)/den)
  if abs(beta)>=abs(observed):extreme+=1
 return {'observed_beta':observed,'permutation_two_sided_p':(extreme+1)/(reps+1),'permutations':reps,
         'null_question':'within-family association of elapsed time with earliest-reference E5 distance'}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--permutations',type=int,default=199)
 ap.add_argument('--seed',type=int,default=13)
 a=ap.parse_args()
 if not EMBED.exists():raise FileNotFoundError(f'Missing original E5 embeddings: {EMBED}')
 E=np.load(EMBED,mmap_mode='r');OUT.mkdir(parents=True,exist_ok=True)
 rows=[];nulls={}
 for variant in ('baseline','deboilerplated'):
  path=INPUT/f'{variant}_version_distances.parquet'
  if not path.exists():raise FileNotFoundError(f'Run Script 42 first: missing {path}')
  d=pd.read_parquet(path).sort_values(['lineage','t_days','text_id']).reset_index(drop=True)
  d['prev_id']=d.groupby('lineage').text_id.shift(1).fillna(d.first_id).astype(np.int64)
  d['prev_time']=d.groupby('lineage').t_days.shift(1).fillna(0)
  d['delta_days']=(d.t_days-d.prev_time).clip(lower=0)
  d['consecutive_distance']=cos_dist(E,d.text_id.to_numpy(dtype=np.int64),d.prev_id.to_numpy(dtype=np.int64))
  d['angular_distance']=np.arccos(np.clip(1-d.distance,-1,1))/np.pi
  d['log_t']=np.log1p(d.t_days.clip(lower=0))
  d['log_delta']=np.log1p(d.delta_days.clip(lower=0))
  for name,x,y in [('earliest_cosine','log_t','distance'),
                   ('consecutive_cosine','log_delta','consecutive_distance'),
                   ('earliest_angular','log_t','angular_distance')]:
   result=fe(d,x,y);rows.append({'variant':variant,'specification':name,**result})
   print('[REGRESSION]',variant,name,result,flush=True)
  if a.permutations>0:
   nulls[variant]=permute_null(d,a.permutations,a.seed)
  d[['lineage','text_id','prev_id','t_days','delta_days','distance','consecutive_distance','angular_distance']].to_csv(OUT/f'{variant}_reference_distances.csv',index=False)
 pd.DataFrame(rows).to_csv(OUT/'temporal_reference_comparison.csv',index=False)
 (OUT/'permutation_nulls.json').write_text(json.dumps(nulls,indent=2),encoding='utf-8')
 print('[DONE]',OUT,flush=True)
if __name__=='__main__':main()
