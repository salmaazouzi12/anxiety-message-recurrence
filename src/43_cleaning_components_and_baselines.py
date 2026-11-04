#!/usr/bin/env python3
"""FAST, pair-level reviewer ablations and classic deduplication baselines.

Tests URL-only, hashtag-only, footer-only and combined cleaning separately;
HashingVectorizer character 3–5 gram cosine, 5-char Jaccard, and token SimHash.
All 400 labels are DEVELOPMENT / stratified, NOT independent gold labels.
These are DIRECT PAIR SCORES, not reconstruction experiments: graph effects and
candidate retrieval differences require separate runs, not implied here.

Usage: python src/43_cleaning_components_and_baselines.py
       python src/43_cleaning_components_and_baselines.py --skip-footer
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs'/'anxiety_lineages';OUT=BASE/'reviewer_experiments'/'pair_ablations'
URL=re.compile(r'(?i)(?:https?://|www\.)[^\s]+')
HASHTAG=re.compile(r'(?<![\w])#[\w\u0600-\u06ff]+',flags=re.UNICODE)
FOOTER=re.compile(r'(?i)(?:follow (?:us|me)|subscribe|visit (?:our|my)|click (?:here|the link)|for more (?:info|information|updates)|all rights reserved|disclaimer|contact us)')

def location(file):
 for p in (ROOT/'src'/file,ROOT/file,BASE/'stage4_full_ablation'/file):
  if p.exists():return p
 raise FileNotFoundError(f'{file} missing. Put it beside this script in src/.')

def footer_catalog(mapping):
 counts=Counter()
 # Exactly the same definition as Script 38: last line of 15–180 chars, target words, >=50 distinct versions.
 for v in mapping.normalized_text.fillna('').astype(str):
  lines=v.splitlines()
  if len(lines)<2:continue
  tail=lines[-1].strip()
  if 15<=len(tail)<=180 and FOOTER.search(tail):counts[tail.lower()]+=1
 return {s for s,n in counts.items() if n>=50}

def clean(v,what,footers):
 v=str(v) if pd.notna(v) else ''
 if what in ('urls','all'):v=URL.sub(' ',v)
 if what in ('hashtags','all'):v=HASHTAG.sub(' ',v)
 if what in ('footers','all'):
  lines=v.splitlines()
  if len(lines)>1 and lines[-1].strip().lower() in footers:v='\n'.join(lines[:-1])
 return ' '.join(v.split())

def cosine_scores(a,b):
 alltexts=list(a)+list(b);v=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**18,alternate_sign=False,norm='l2',dtype=np.float32)
 x=v.transform(alltexts);n=len(a)
 return np.asarray(x[:n].multiply(x[n:]).sum(axis=1)).ravel()

def lexical_normalize(t):
 t=t.lower();t=re.sub(r'https?://\S+',' ',t);t=t.replace('ـ','')
 return ' '.join(re.sub(r'[^\w\s]',' ',t,flags=re.UNICODE).split())

def shingles(s,n=5):
 t=lexical_normalize(s)
 return set(t[i:i+n] for i in range(len(t)-n+1)) if len(t)>=n else ({t} if t else set())

def simhash64(text):
 # Deterministic token SimHash, unlike Python's salted built-in hash.
 tokens=lexical_normalize(text).split();c=Counter(tokens)
 vec=np.zeros(64,dtype=np.int32)
 for tok,w in c.items():
  h=int.from_bytes(hashlib.blake2b(tok.encode('utf-8'),digest_size=8).digest(),'big')
  for i in range(64):vec[i]+=w if (h>>i)&1 else -w
 return sum((1<<i) for i,x in enumerate(vec) if x>0)

def fixed_scores(a,b):
 sj=[];ss=[]
 for x,y in zip(a,b):
  u=shingles(x);v=shingles(y)
  sj.append(len(u&v)/len(u|v) if (u|v) else 1.0)
  ha,hb=simhash64(x),simhash64(y)
  ss.append(1-(ha^hb).bit_count()/64)
 return np.array(sj),np.array(ss)

def groups_from_family_ids(df):
 # Prevent family leakage: union any two original baseline family names occurring in a pair.
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  ra,rb=find(a),find(b)
  if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
 for r in df.itertuples(index=False):union(str(r.lineage_a),str(r.lineage_b))
 return np.array([find(str(x)) for x in df.lineage_a])

def report(y,score,train,test):
 # Choose direct-score threshold ONLY on the DEVELOPMENT training partition.
 # The holdout is family-disjoint within the *same* reused labeled development set;
 # it is not an independently collected benchmark.
 cuts=np.unique(np.r_[0.0,0.45,0.60,0.70,0.75,0.80,0.83,0.86,0.88,0.90,0.92,0.94,0.96,0.98,1.0,np.quantile(score[train],np.linspace(0,1,51))])
 best=None
 for t in cuts:
  pred=score[train]>=t
  p,r,f,_=precision_recall_fscore_support(y[train],pred,average='binary',zero_division=0)
  if best is None or (f,p)> (best['train_f1'],best['train_precision']):best={'threshold':float(t),'train_f1':float(f),'train_precision':float(p)}
 pp=score[test]>=best['threshold']
 p,r,f,_=precision_recall_fscore_support(y[test],pp,average='binary',zero_division=0)
 result={**best,'n_train':int(len(train)),'n_test':int(len(test)),'test_precision':float(p),'test_recall':float(r),'test_f1':float(f),
         'test_tp':int(((pp==1)&(y[test]==1)).sum()),'test_fp':int(((pp==1)&(y[test]==0)).sum()),
         'test_fn':int(((pp==0)&(y[test]==1)).sum()),'test_tn':int(((pp==0)&(y[test]==0)).sum())}
 if len(np.unique(y[test]))>1:
  result['test_roc_auc']=float(roc_auc_score(y[test],score[test]));result['test_average_precision']=float(average_precision_score(y[test],score[test]))
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--skip-footer',action='store_true',help='Only pairwise URL, hashtag and combined-without-footer ablations; not equivalent to full Script 38 cleaning')
 ap.add_argument('--seed',type=int,default=42)
 args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 ann=pd.read_csv(location('pair_annotations_completed.csv'))
 key=pd.read_csv(location('pair_sampling_key.csv'))
 df=key.merge(ann[['pair_id','text_a','text_b','label','annotation_origin']],on='pair_id',validate='one_to_one')
 if len(df)!=400:raise ValueError(f'Expected 400 joined pairs, got {len(df)}')
 y=df.label.eq('SAME').to_numpy(dtype=np.int8)
 catalog=set();mapping=BASE/'minhash'/'text_id_mapping.parquet'
 if not args.skip_footer:
  if not mapping.exists():raise FileNotFoundError(f'Need original mapping for faithful footer frequency: {mapping}. Or --skip-footer')
  catalog=footer_catalog(pd.read_parquet(mapping,columns=['normalized_text']))
  print(f'[FOOTERS] {len(catalog)} qualifying footer strings',flush=True)
 methods=['raw','urls','hashtags','footers','all'] if not args.skip_footer else ['raw','urls','hashtags','urls_and_hashtags']
 if args.skip_footer:print('[CAUTION] No full-cleaning comparison: footer catalog unavailable',flush=True)
 scores={};texts={}
 for m in methods:
  which='all' if m=='urls_and_hashtags' else m
  aa=[clean(t,which,catalog) for t in df.text_a]
  bb=[clean(t,which,catalog) for t in df.text_b]
  scores['char_cosine_'+m]=cosine_scores(aa,bb)
  texts[m]=(aa,bb)
 jaccard,simhash=fixed_scores(*texts['raw'])
 scores['shingle_jaccard_raw']=jaccard;scores['token_simhash_raw']=simhash
 prediction=pd.DataFrame({'pair_id':df.pair_id,'label':df.label,'sampling_stratum':df.sampling_stratum,'annotation_origin':df.annotation_origin})
 for k,score in scores.items():prediction[k]=score
 prediction.to_csv(OUT/'pair_similarity_scores.csv',index=False)
 groups=groups_from_family_ids(df);spl=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=args.seed)
 tr,te=next(spl.split(np.zeros(len(y)),y,groups))
 if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2:raise ValueError('The family-disjoint split lacks both classes; change --seed')
 rows=[]
 for k,s in scores.items():
  r=report(y,s,tr,te);r['method']=k
  rows.append(r)
  print('[BENCHMARK]',k,'threshold',round(r['threshold'],4),'test F1',round(r['test_f1'],3),flush=True)
 out=pd.DataFrame(rows).sort_values('test_f1',ascending=False)
 out.to_csv(OUT/'development_family_disjoint_benchmarks.csv',index=False)
 (OUT/'study_design.json').write_text(json.dumps({'development_pairs':400,'train_pairs':len(tr),'test_pairs':len(te),'split_seed':args.seed,
  'connected_original_family_groups':int(len(np.unique(groups))), 'train_family_groups':len(np.unique(groups[tr])),
  'test_family_groups':len(np.unique(groups[te])),'note':'Thresholds selected on training part of reused, stratified annotation sample. This is NOT an independent, blinded gold standard. Pair-only direct scores, not full reconstruction.'},indent=2),encoding='utf-8')
 print('[DONE]',OUT,flush=True)
if __name__=='__main__':main()
