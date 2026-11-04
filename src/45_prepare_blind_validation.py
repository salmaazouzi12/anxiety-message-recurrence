#!/usr/bin/env python3
"""Prepare NEW blinded, family-disjoint pair and family annotation sheets.

No LLM annotates anything. Two humans independently label the same shuffled
items; the private key is kept out of annotator files. Excludes every family
appearing in the original 400-pair development audit and avoids reusing any
baseline OR cleaned family across selected items.

python src/45_prepare_blind_validation.py --pairs 120 --families 40 --seed 20261101
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs'/'anxiety_lineages';OUT=BASE/'reviewer_experiments'/'blind_validation'
MAP=BASE/'minhash'/'text_id_mapping.parquet'
BASE_MEM=BASE/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
CLEAN_MEM=BASE/'stage4_full_ablation'/'deboilerplated'/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
OLD_KEY_CANDIDATES=[ROOT/'src'/'pair_sampling_key.csv',ROOT/'pair_sampling_key.csv']

def mem(path):
 if not path.exists():raise FileNotFoundError(path)
 d=pd.read_parquet(path,columns=['text_id','repaired_lineage_id'])
 if d.text_id.duplicated().any():raise ValueError(f'Duplicate membership IDs: {path}')
 return dict(zip(d.text_id.astype(int),d.repaired_lineage_id.astype(int)))

def candidate_stratum(a,b,base,clean):
 bs=a in base and b in base and base[a]==base[b]
 cs=a in clean and b in clean and clean[a]==clean[b]
 if bs and cs:return 'stable_same'
 if bs and not cs:return 'split_by_cleaning'
 if (not bs) and cs:return 'newly_linked'
 return 'both_separate'

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--pairs',type=int,default=120)
 ap.add_argument('--families',type=int,default=40)
 ap.add_argument('--seed',type=int,default=20261101)
 a=ap.parse_args()
 rng=np.random.default_rng(a.seed);OUT.mkdir(parents=True,exist_ok=True)
 original=next((p for p in OLD_KEY_CANDIDATES if p.exists()),None)
 if original is None:raise FileNotFoundError('Need original pair_sampling_key.csv next to script in src/ to exclude all known development families')
 old=pd.read_csv(original)
 mapping=pd.read_parquet(MAP,columns=['text_id','normalized_text','post_count'])
 mapping=mapping.set_index('text_id')
 base=mem(BASE_MEM);clean=mem(CLEAN_MEM)
 already=set(map(int,pd.concat([old.text_id_a,old.text_id_b],ignore_index=True)))
 # Ban WHOLE families of all previously sampled development pair text IDs.
 banned_base={base[t] for t in already if t in base}
 banned_clean={clean[t] for t in already if t in clean}
 used_base=set(banned_base);used_clean=set(banned_clean);used_text=set(already)
 by_base=defaultdict(list);by_clean=defaultdict(list)
 for t,root in base.items():
  if root not in banned_base and t not in already:by_base[root].append(t)
 for t,root in clean.items():
  if root not in banned_clean and t not in already:by_clean[root].append(t)
 def available_ids(ids):
  for t in ids:
   if t in used_text or base.get(t) in used_base or clean.get(t) in used_clean:return False
  return True
 def claim(ids):
  used_text.update(ids)
  used_base.update(base[t] for t in ids if t in base)
  used_clean.update(clean[t] for t in ids if t in clean)
 # Create a small pool from EACH group; iterate on demand, not all O(n^2) pairs.
 pools=defaultdict(list)
 for root,items in by_base.items():
  if len(items)<2:continue
  items=list(items)
  for _ in range(min(8,len(items)*2)):
   ids=rng.choice(items,size=2,replace=False);x,y=map(int,ids)
   pools[candidate_stratum(x,y,base,clean)].append((x,y,'baseline_family'))
 for root,items in by_clean.items():
  if len(items)<2:continue
  for _ in range(min(8,len(items)*2)):
   ids=rng.choice(items,size=2,replace=False);x,y=map(int,ids)
   pools[candidate_stratum(x,y,base,clean)].append((x,y,'cleaned_family'))
 # Add semantic/lexical candidates that do not belong to a common linked family.
 sem=BASE/'combined_retrieval'/'e5_novel_pairs.parquet'
 if sem.exists():
  import pyarrow.parquet as pq
  pf=pq.ParquetFile(sem);columns=pf.schema_arrow.names
  if {'text_id_1','text_id_2'}.issubset(columns):
   # Reservoir for close candidates (ranking is not used to label the pool).
   reservoir=[];total=0;cap=3000
   for chunk in pf.iter_batches(batch_size=80000,columns=['text_id_1','text_id_2']):
    d=chunk.to_pydict()
    for u,v in zip(d['text_id_1'],d['text_id_2']):
     u,v=int(u),int(v)
     if u in already or v in already or base.get(u) in banned_base or base.get(v) in banned_base or clean.get(u) in banned_clean or clean.get(v) in banned_clean:continue
     if candidate_stratum(u,v,base,clean)!='both_separate':continue
     total+=1
     if len(reservoir)<cap:reservoir.append((u,v,'novel_e5_retrieved'))
     else:
      j=int(rng.integers(0,total))
      if j<cap:reservoir[j]=(u,v,'novel_e5_retrieved')
   pools['near_candidate_separate'].extend(reservoir)
   print('[CANDIDATE POOL] eligible rejected semantic candidates:',total,flush=True)
 # Random different-family comparison controls.
 eligible=np.array([t for t in mapping.index if t not in already and base.get(t) not in banned_base and clean.get(t) not in banned_clean],dtype=np.int64)
 for _ in range(min(2000,len(eligible))):
  u,v=map(int,rng.choice(eligible,size=2,replace=False))
  if candidate_stratum(u,v,base,clean)=='both_separate':pools['random_different'].append((u,v,'random_control'))
 targets={'stable_same':int(a.pairs*.25),'split_by_cleaning':int(a.pairs*.20),
          'newly_linked':int(a.pairs*.10),'near_candidate_separate':int(a.pairs*.30)}
 targets['random_different']=a.pairs-sum(targets.values())
 chosen=[]
 def take(k,n):
  pool=pools[k]
  if len(pool):
   for i in rng.permutation(len(pool)):
    u,v,source=pool[int(i)]
    if u==v or u not in mapping.index or v not in mapping.index or not available_ids((u,v)):continue
    claim((u,v));chosen.append({'text_id_a':u,'text_id_b':v,'stratum':k,'source':source})
    n-=1
    if n<=0:break
  return n
 for k,n in targets.items():
  rem=take(k,n)
  if rem:print(f'[NOTE] {k}: requested {n}, short by {rem} under strict family disjointness',flush=True)
 if len(chosen)<a.pairs:
  for k in ('stable_same','near_candidate_separate','random_different','split_by_cleaning','newly_linked'):
   if len(chosen)>=a.pairs:break
   take(k,a.pairs-len(chosen))
 # Family-triplet audit, from DISTINCT families not seen in old or new pair sample.
 fam=[]
 possibilities=[(root,ids) for root,ids in by_clean.items() if len(ids)>=3]
 rng.shuffle(possibilities)
 for root,ids in possibilities:
  ids=rng.choice(ids,size=3,replace=False).tolist()
  ids=list(map(int,ids))
  if not available_ids(ids):continue
  claim(ids)
  fam.append({'ids':ids,'clean_family':root})
  if len(fam)>=a.families:break
 # Ensure same items but RANDOMIZED row orders in each annotator's sheet.
 key=[];items=[]
 for i,r in enumerate(chosen,1):
  pid=f'H{i:04d}';u=int(r['text_id_a']);v=int(r['text_id_b'])
  if rng.random()<.5:u,v=v,u
  items.append({'item_id':pid,'text_a':str(mapping.loc[u,'normalized_text']),'text_b':str(mapping.loc[v,'normalized_text']),
                'label':'','confidence':'','notes':''})
  key.append({'item_id':pid,'text_id_a':u,'text_id_b':v,'sampling_stratum':r['stratum'],'baseline_same':int(candidate_stratum(u,v,base,base)=='stable_same'),
              'cleaned_same':int(candidate_stratum(u,v,clean,clean)=='stable_same')})
 fkey=[];fitems=[]
 for i,r in enumerate(fam,1):
  fid=f'F{i:04d}';ids=r['ids']
  fitems.append({'item_id':fid,'text_1':str(mapping.loc[ids[0],'normalized_text']),
                 'text_2':str(mapping.loc[ids[1],'normalized_text']),
                 'text_3':str(mapping.loc[ids[2],'normalized_text']),
                 'label':'','confidence':'','notes':''})
  fkey.append({'item_id':fid,'text_ids':','.join(map(str,ids)),'cleaned_lineage_id':int(r['clean_family'])})
 def write_annotators(prefix,rows):
  for name in ('A','B'):
   order=rng.permutation(len(rows));pd.DataFrame([rows[int(i)] for i in order]).to_csv(OUT/f'{prefix}_annotator_{name}.csv',index=False,encoding='utf-8-sig')
 if items:write_annotators('blinded_pairs',items)
 if fitems:write_annotators('blinded_families',fitems)
 pd.DataFrame(key).to_csv(OUT/'PRIVATE_pair_sampling_key_DO_NOT_SHARE.csv',index=False)
 pd.DataFrame(fkey).to_csv(OUT/'PRIVATE_family_sampling_key_DO_NOT_SHARE.csv',index=False)
 protocol='''# Independent blinded annotation protocol\n\nEach of two HUMAN annotators must work independently WITHOUT seeing the PRIVATE keys, family assignments, model predictions or the other annotator's labels. Do not prefill labels with AI predictions.\n\n**Pairs:** SAME = substantially the same circulating message despite small edits; RELATED = topic/keywords shared but different core message; UNRELATED = no core message identity. For uncertainty, use confidence LOW and explain why, but assign a best label.\n\n**Three-version families:** COHERENT = all 3 share core message; PARTIALLY COHERENT = only some share it; NOT COHERENT = substantive messages differ; NOT SURE = unable to judge.\n\nRecord notes, confidence (HIGH/MEDIUM/LOW). Afterwards compare labels blind, adjudicate disagreements without seeing algorithm predictions; report both original agreement and adjudicated results. The PRIVATE key must not be given to annotators.\n\nThese new sampled families are family-disjoint from development samples by original and cleaned reconstructed families, insofar as the graph memberships expose their IDs. These are not probability draws from all possible Facebook-text pairs.\n'''
 (OUT/'ANNOTATION_PROTOCOL.md').write_text(protocol,encoding='utf-8')
 meta={'requested_pair_items':a.pairs,'exported_pair_items':len(items),'requested_family_items':a.families,'exported_family_items':len(fitems),
       'stratum_counts':{str(k):int(v) for k,v in pd.Series([r['stratum'] for r in chosen]).value_counts().items()},'seed':int(a.seed),
       'note':'No human labels generated. Must use two independent human annotators.'}
 (OUT/'sample_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print('[DONE] generated',len(items),'blinded pair items and',len(fitems),'family triplets at',OUT,flush=True)
if __name__=='__main__':main()
