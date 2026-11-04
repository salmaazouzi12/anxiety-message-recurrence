#!/usr/bin/env python3
"""Re-run the ORIGINAL MinHash LSH retrieval on original vs cleaned text IDs.

Fast diagnostic by default: 5,000 random text versions PLUS all 400 annotated
pair endpoints. For selected pairs, uses the same 64-permutation, 5-character
shingle MinHash and LSH threshold 0.45 as src/27_minhash_candidates.py.

This compares LEXICAL retrieval only; not complete E5/FAISS re-retrieval.
Use --full for all 253,983 texts (expensive; no fast turnaround promised).
python src/48_cleaned_minhash_retrieval_audit.py --sample 5000
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs'/'anxiety_lineages';OUT=BASE/'reviewer_experiments'/'minhash_retrieval_shift'

def lex(text):
 text=text.lower();text=re.sub(r'https?://\S+',' ',text);text=text.replace('ـ','')
 return ' '.join(re.sub(r'[^\w\s]',' ',text,flags=re.UNICODE).split())

def shingles(s):
 s=lex(s)
 if not s:return {b'<EMPTY>'}
 if len(s)<5:return {s.encode('utf-8')}
 return {s[i:i+5].encode('utf-8') for i in range(len(s)-4)}

def retrieve(ids,texts):
 from datasketch import MinHash, MinHashLSH
 lsh=MinHashLSH(threshold=0.45,num_perm=64)
 found=set()
 for counter,i in enumerate(ids,1):
  mh=MinHash(num_perm=64,seed=42)
  for gram in shingles(texts[i]):mh.update(gram)
  for other in lsh.query(mh):
   j=int(other)
   if i!=j:found.add((min(i,j),max(i,j)))
  lsh.insert(str(i),mh)
  if counter%2000==0:print(f'[LSH] {counter:,}/{len(ids):,} unique versions; {len(found):,} candidate pairs',flush=True)
 return found

def main():
 try:import datasketch
 except ImportError:raise SystemExit('Install datasketch: python -m pip install datasketch')
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--sample',type=int,default=5000)
 ap.add_argument('--full',action='store_true',help='Run on all mapped texts, potentially expensive')
 ap.add_argument('--seed',type=int,default=44)
 a=ap.parse_args()
 orig=BASE/'minhash'/'text_id_mapping.parquet'
 cleaned=BASE/'stage4_full_ablation'/'deboilerplated'/'text_id_mapping_clean.parquet'
 if not orig.exists() or not cleaned.exists():raise FileNotFoundError('Need original and Script40 de-boilerplated normalized text mappings')
 o=pd.read_parquet(orig,columns=['text_id','normalized_text']).sort_values('text_id').reset_index(drop=True)
 c=pd.read_parquet(cleaned,columns=['text_id','normalized_text']).sort_values('text_id').reset_index(drop=True)
 if not np.array_equal(o.text_id,c.text_id):raise ValueError('Original and cleaned ID mappings disagree')
 n=len(o);rng=np.random.default_rng(a.seed)
 ids=np.arange(n,dtype=np.int64) if a.full else np.sort(rng.choice(n,size=min(a.sample,n),replace=False))
 key=next((p for p in (ROOT/'src'/'pair_sampling_key.csv',ROOT/'pair_sampling_key.csv') if p.exists()),None)
 pairs=None
 if key:
  pairs=pd.read_csv(key)
  ids=np.unique(np.r_[ids,pairs.text_id_a.to_numpy(dtype=np.int64),pairs.text_id_b.to_numpy(dtype=np.int64)])
 print('[RUN] original MinHash over',len(ids),'fixed IDs',flush=True)
 raw=retrieve(ids,o.normalized_text.fillna('').astype(str).tolist())
 print('[RUN] cleaned MinHash over same IDs',flush=True)
 new=retrieve(ids,c.normalized_text.fillna('').astype(str).tolist())
 OUT.mkdir(parents=True,exist_ok=True)
 print('[RESULT] original',len(raw),'cleaned',len(new),'new_only',len(new-raw),'lost',len(raw-new),flush=True)
 result={'n_selected_text_ids':len(ids),'original_lexical_candidates':len(raw),'cleaned_lexical_candidates':len(new),
         'new_only_candidate_pairs':len(new-raw),'original_only_candidate_pairs':len(raw-new),
         'comparison':'MinHash lexical candidate retrieval on same fixed subset of IDs; E5 not rerun',
         'sample':'Full corpus' if a.full else f'{a.sample} random text IDs + all original annotated pair endpoints'}
 if pairs is not None:
  labels=next((p for p in (ROOT/'src'/'pair_annotations_completed.csv',ROOT/'pair_annotations_completed.csv') if p.exists()),None)
  if labels:
   pairs=pairs.merge(pd.read_csv(labels)[['pair_id','label']],on='pair_id',validate='one_to_one')
  rows=[]
  for r in pairs.itertuples(index=False):
   x,y=sorted((int(r.text_id_a),int(r.text_id_b)))
   rows.append({'pair_id':r.pair_id,'label':getattr(r,'label',''),
                'original_minhash_retrieved':int((x,y) in raw),
                'cleaned_minhash_retrieved':int((x,y) in new)})
  pd.DataFrame(rows).to_csv(OUT/'annotated_pair_retrieval.csv',index=False)
  if labels:
   p=pd.DataFrame(rows);positive=p[p.label=='SAME']
   result['same_annotated_pairs']=len(positive)
   result['minhash_retrieval_rate_same_original']=float(positive.original_minhash_retrieved.mean())
   result['minhash_retrieval_rate_same_cleaned']=float(positive.cleaned_minhash_retrieved.mean())
 (OUT/'retrieval_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('[DONE]',OUT,flush=True)
if __name__=='__main__':main()
