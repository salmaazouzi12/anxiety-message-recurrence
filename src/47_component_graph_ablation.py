#!/usr/bin/env python3
"""Resumable ISOLATED full graph reconstruction for ONE cleaning component at a time.

Actual Scripts 34 and 35 are reused; the original candidate pool is fixed.
Each component entails a separate vector cache + scored candidates + rebuild.
This is expensive; run hashtags_only first if pair-level evidence supports it.

python src/47_component_graph_ablation.py --variant hashtags_only --quiet
python src/47_component_graph_ablation.py --variant urls_only --quiet
python src/47_component_graph_ablation.py --variant footers_only --quiet

Requires 38_stage4_faithful_ablation.py, 40_stage4_boilerplate_fast.py,
original Scripts 34/35, and the existing candidate files. Outputs in isolated
stage4_full_ablation/<variant> directories; never overwrites other variants.
"""
from __future__ import annotations
import argparse, importlib.util, json, re, os, sys, csv
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'src'

def module(filename,name):
 p=SRC/filename
 if not p.exists():raise FileNotFoundError(p)
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def create_mapping(st,fast,variant):
 import pandas as pd
 src=st.BASE/'minhash'/'text_id_mapping.parquet'
 dest=st.OUT/variant/'text_id_mapping_clean.parquet'
 report=dest.parent/'text_cleaning_summary.json'
 signature=fast.sig(src)
 if dest.exists() and report.exists():
  prior=json.loads(report.read_text(encoding='utf-8'))
  if prior.get('source_signature')==signature and prior.get('variant')==variant:
   d=pd.read_parquet(dest,columns=['normalized_text']);print('[REUSE] Existing variant mapping',variant,flush=True)
   return d.normalized_text.fillna('').astype(str).tolist(),dest
 d=pd.read_parquet(src).sort_values('text_id').reset_index(drop=True)
 if not np.array_equal(d.text_id.to_numpy(),np.arange(len(d))):raise ValueError('Expected contiguous IDs, as in original Script 34')
 texts=d.normalized_text.fillna('').astype(str).tolist()
 URL=re.compile(r'(?i)(?:https?://|www\.)[^\s]+')
 HASH=re.compile(r'(?<![\w])#[\w\u0600-\u06ff]+',flags=re.UNICODE)
 FOOT=re.compile(r'(?i)(?:follow (?:us|me)|subscribe|visit (?:our|my)|click (?:here|the link)|for more (?:info|information|updates)|all rights reserved|disclaimer|contact us)')
 repeated=set()
 if variant=='footers_only':
  counts=Counter()
  for text in texts:
   lines=text.splitlines()
   if len(lines)>1:
    tail=lines[-1].strip()
    if 15<=len(tail)<=180 and FOOT.search(tail):counts[tail.lower()]+=1
  repeated={t for t,c in counts.items() if c>=50}
 altered=0;remove_footers=0;cleaned=[]
 for text in texts:
  v=text
  if variant=='urls_only':v=URL.sub(' ',v)
  elif variant=='hashtags_only':v=HASH.sub(' ',v)
  else:
   lines=v.splitlines()
   if len(lines)>1 and lines[-1].strip().lower() in repeated:
    v='\n'.join(lines[:-1]);remove_footers+=1
  v=' '.join(v.split())
  if v!=text:altered+=1
  cleaned.append(v)
 d['normalized_text']=cleaned
 dest.parent.mkdir(parents=True,exist_ok=True)
 d.to_parquet(dest,index=False)
 fast.atomic_json(report,{'source_signature':signature,'variant':variant,'texts':len(d),
                          'texts_changed':altered,'footer_types':len(repeated),'footer_occurrences_removed':remove_footers,
                          'operations':'Only the single specified operation, whitespace collapse, original text IDs and post counts retained'})
 print('[CLEAN]',variant,'modified texts=',altered,'footer types=',len(repeated),flush=True)
 return cleaned,dest

def score(st,fast,variant,X,clean_mapping,batch_size):
 import pyarrow.parquet as pq
 candidates=[('minhash',sorted((st.BASE/'minhash').glob('candidate_pairs_*.parquet')),
              st.OUT/variant/'lexical_scoring','scored_pairs'),
             ('e5',[st.BASE/'combined_retrieval'/'e5_novel_pairs.parquet'],
              st.OUT/variant/'combined_retrieval'/'novel_e5_scored','scored_semantic')]
 clean_sig=fast.sig(clean_mapping)
 for name,inputs,out,prefix in candidates:
  if not inputs or any(not p.exists() for p in inputs):raise FileNotFoundError(f'Missing original candidate files for {name}')
  out.mkdir(parents=True,exist_ok=True)
  n_done=n_fresh=0;expected=set()
  for j,p in enumerate(inputs,1):
   pf=pq.ParquetFile(p)
   if not {'text_id_1','text_id_2'}.issubset(pf.schema_arrow.names):raise ValueError(f'Unexpected candidate schema in {p}')
   source_sig=fast.sig(p)
   for k,batch in enumerate(pf.iter_batches(columns=['text_id_1','text_id_2'],batch_size=batch_size)):
    path=out/f'{prefix}_fast_{j:05d}_{k:06d}.parquet';mark=path.with_name(path.name+'.done.json');expected.add(path)
    ck={'source':source_sig,'clean_mapping':clean_sig,'batch_index':k,'batch_size':batch_size,'batch_rows':batch.num_rows,
        'variant':variant,'algorithm':'hash-char3-5,nonnegative,2^18,L2,threshold0.86'}
    valid=False
    if path.exists() and mark.exists():
     try:valid=all(json.loads(mark.read_text(encoding='utf-8')).get(z)==v for z,v in ck.items()) and fast.validate_scored(path)
     except (OSError,ValueError):pass
    if valid:n_done+=1;continue
    d=batch.to_pydict();u=np.asarray(d['text_id_1'],dtype=np.int64);v=np.asarray(d['text_id_2'],dtype=np.int64)
    if ((u<0)|(u>=X.shape[0])|(v<0)|(v>=X.shape[0])).any():raise ValueError(f'Bad candidate text ID {p}')
    sims=fast.score_batch(X,u,v)
    fast.write_parquet_atomic(path,u,v,sims)
    fast.atomic_json(mark,ck);n_fresh+=1
    if n_fresh%25==0:print('[SCORE]',variant,name,'newly done',n_fresh,flush=True)
  for p in out.glob(f'{prefix}_*.parquet'):
   if p not in expected:
    # Never allow stale score shards in the active folder to contaminate Script 34.
    archive=out/'archived_unverified';archive.mkdir(exist_ok=True)
    os.replace(p,archive/p.name)
    mark=p.with_name(p.name+'.done.json')
    if mark.exists():os.replace(mark,archive/mark.name)
  print('[DONE]',name,'reused',n_done,'new',n_fresh,'score shards',flush=True)

def find_pair_labels():
    """Find 400-pair development labels and key by their content, not hard-coded paths.

    This is called before any expensive vectorization. Never generates or
    modifies human annotations or sampling keys.
    """
    required_key = {'pair_id', 'text_id_a', 'text_id_b', 'source_group'}
    required_lab = {'pair_id', 'label'}
    def find(glob, required):
        found=[]
        # Search the familiar source/output locations and nested subfolders.
        for parent in (ROOT / 'src', ROOT / 'outputs', ROOT):
            if not parent.exists():continue
            for path in parent.rglob(glob):
                if not path.is_file() or path in found:continue
                try:
                    with path.open('r', encoding='utf-8-sig', newline='') as f:
                        rows=list(csv.DictReader(f))
                    if len(rows)==400 and rows and required.issubset(rows[0]):
                        found.append(path)
                except (OSError, UnicodeError, csv.Error):continue
        return found
    keys=find('pair_sampling_key*.csv', required_key)
    labs=find('pair_annotations_completed*.csv', required_lab)
    if not keys or not labs:
        raise FileNotFoundError('Cannot locate the 400-pair development key and labels under src/ or outputs/. '
                                f'Found {len(keys)} compatible key(s), {len(labs)} compatible label file(s). '
                                'Do not rerun sampling.')
    for key in keys:
        with key.open('r',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        keyids={r['pair_id'] for r in rows}
        if len(keyids)!=400:continue
        for lab in labs:
            with lab.open('r',encoding='utf-8-sig',newline='') as f:ann=list(csv.DictReader(f))
            byid={r['pair_id']:r for r in ann}
            if len(byid)!=400 or keyids!=set(byid):continue
            out=[]
            for k in rows:
                v=byid[k['pair_id']]
                gold=v['label'].strip().upper()
                if gold not in {'SAME','RELATED','UNRELATED'}:
                    raise ValueError(f'Invalid reference label {gold!r} for {k["pair_id"]}')
                out.append({'pair_id':k['pair_id'],'a':int(k['text_id_a']), 'b':int(k['text_id_b']),
                            'gold':gold,'source_group':k.get('source_group',''),
                            'sampling_stratum':k.get('sampling_stratum',''),
                            'annotation_origin':v.get('annotation_origin',''),
                            'original_sampling_decision':int(k.get('source_group','')=='same_reconstructed_family')})
            print(f'[LABELS] 400 development pairs verified: {key.relative_to(ROOT)}; {lab.relative_to(ROOT)}',flush=True)
            return out
    raise ValueError('Found candidate key/label files but no matching set of 400 pair IDs.')


def preflight(st,fast):
    """Validate all inputs BEFORE any costly graph run."""
    labels=find_pair_labels()
    st.read_orig('34_build_combined_lineages.py')
    st.read_orig('35_repair_combined_lineages.py')
    sources=[st.BASE/'minhash'/'text_id_mapping.parquet',
             st.BASE/'combined_retrieval'/'e5_novel_pairs.parquet']
    paths=list((st.BASE/'minhash').glob('candidate_pairs_*.parquet'))
    if not paths:raise FileNotFoundError('No MinHash candidate pair shards found under outputs/anxiety_lineages/minhash/')
    for p in sources:
        if not p.is_file():raise FileNotFoundError(f'Required existing candidate input missing: {p}')
    print(f'[PREFLIGHT] Original graph scripts and candidate pool verified; {len(paths)} MinHash shards',flush=True)
    return labels


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--variant',choices=['urls_only','hashtags_only','footers_only'],required=True)
 ap.add_argument('--quiet',action='store_true')
 ap.add_argument('--batch-size',type=int,default=6000)
 ap.add_argument('--vector-chunk',type=int,default=2000)
 ap.add_argument('--check',action='store_true',help='Validate required inputs without any expensive work')
 a=ap.parse_args()
 if a.batch_size<100 or a.vector_chunk<100:raise ValueError('batch-size and vector-chunk must be >=100')
 st=module('38_stage4_faithful_ablation.py','s38');fast=module('40_stage4_boilerplate_fast.py','s40')
 st.OUT.mkdir(parents=True,exist_ok=True);st.GENERATED.mkdir(parents=True,exist_ok=True)
 labels=preflight(st,fast)
 if a.check:
  print('[CHECK COMPLETE] No vectors scored, no graph rebuilt.',flush=True);return
 texts,cleanpath=create_mapping(st,fast,a.variant)
 X=fast.build_sparse_cache(texts,cleanpath,st.OUT/a.variant/'sparse_vector_cache',a.vector_chunk)
 score(st,fast,a.variant,X,cleanpath,a.batch_size)
 p=st.OUT/a.variant/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
 preliminary=st.OUT/a.variant/'combined_lineages'/'combined_lineage_members.parquet'
 inputs=[q for path,pattern in [(st.OUT/a.variant/'lexical_scoring','scored_pairs_*.parquet'),
      (st.OUT/a.variant/'combined_retrieval'/'novel_e5_scored','scored_semantic_*.parquet')] for q in path.glob(pattern)]
 latest=max(q.stat().st_mtime_ns for q in inputs)
 if not p.exists() or not preliminary.exists() or min(p.stat().st_mtime_ns,preliminary.stat().st_mtime_ns)<latest:
  build=st.generate_build_clean(a.variant);st.run_original_variant(build,verbose=not a.quiet)
  repair=st.generate_repair(a.variant,source=a.variant);st.run_original_variant(repair,verbose=not a.quiet)
 else:print('[REUSE] Graph and repair newer than all scored inputs',flush=True)
 summary,cases=st.evaluate(a.variant,labels,p)
 save=st.BASE/'reviewer_experiments'/'component_graphs';save.mkdir(parents=True,exist_ok=True)
 st.save_rows(save/f'{a.variant}_pair_predictions.csv',cases)
 (save/f'{a.variant}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print('[RESULT]',json.dumps(summary,indent=2),flush=True)
 print('[CAUTION] Original fixed candidate pool, stratified development labels, not held-out gold.',flush=True)
if __name__=='__main__':main()
