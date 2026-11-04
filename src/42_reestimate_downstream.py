#!/usr/bin/env python3
"""Re-estimate family recurrence, cross-page, lifespan and E5 temporal regressions.

Uses EXISTING baseline and de-boilerplated repaired lineage memberships; never
reconstructs or overwrites them. Never silently assumes a post-level join key.

Usage: python src/42_reestimate_downstream.py --posts data/processed/posts_clean.parquet
       python src/42_reestimate_downstream.py --inspect

The script checks post-count parity and warns if baseline temporal estimates do
not reproduce the published values. 'Alternative' results are then exploratory.
"""
from __future__ import annotations
import argparse, json, math, sys, re, unicodedata, html, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs'/'anxiety_lineages'
OUT=BASE/'reviewer_experiments'/'downstream'
MAPPING=BASE/'minhash'/'text_id_mapping.parquet'
EMBED=BASE/'e5_retrieval'/'e5_embeddings.npy'
MEMBERS={
 'baseline':BASE/'combined_cohesion_repair'/'repaired_lineage_members.parquet',
 'deboilerplated':BASE/'stage4_full_ablation'/'deboilerplated'/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
}
POST_CANDIDATES=[
 'data/processed/posts_clean.parquet','data/processed/posts.parquet',
 'data/posts_clean.parquet','outputs/anxiety_lineages/posts_clean.parquet',
 'outputs/anxiety_lineages/posts.parquet','outputs/posts_clean.parquet',
 'data/processed/posts_with_text_ids.parquet'
]
ALIASES={
 'text_id':['text_id','unique_text_id'],
 'exact_family_id':['exact_family_id','exact_text_id'],
 'normalized_text':['normalized_text','text_clean','cleaned_text','message_normalized'],
 'date':['Post Created','post_created_at','created_at','created_time','timestamp','date_time','date','Post Created Date','post_created_date','post_date'],
 'page':['Page Name','page_id','Page ID','page_name','pageId','page','page_identifier'],
 'category':['Page Category','page_category','pageCategory','category'],
 'country':['Page Admin Top Country','page_admin_country','page_admin_top_country','admin_country','country'],
 'post_id':['post_id','Post ID','Facebook Id','facebook_id','id']
}
def alias(cols,key):
 for name in ALIASES[key]:
  if name in cols: return name
 names={str(c).lower().replace(' ','').replace('_',''):c for c in cols}
 for name in ALIASES[key]:
  k=name.lower().replace(' ','').replace('_','')
  if k in names: return names[k]
 return None

def inspect():
 import pyarrow.parquet as pq
 for path in [MAPPING,*MEMBERS.values(),*([EMBED] if EMBED.exists() else [])]:
  print('[FILE]',path, 'FOUND' if path.exists() else 'MISSING')
 for entry in POST_CANDIDATES:
  p=ROOT/entry
  if p.exists():
   print('[POST CANDIDATE]',p,'COLUMNS',pq.ParquetFile(p).schema_arrow.names)
 # Discover other likely post metadata files without opening thousands of scored-edge shards.
 found=[]
 for d in (ROOT/'data',ROOT/'outputs'):
  if d.exists():
   found.extend(p for p in d.rglob('*.parquet') if any(k in p.name.lower() for k in ('post','metadata','cleaned','corpus')))
 for path in sorted(set(found))[:40]:
  if path not in [ROOT/c for c in POST_CANDIDATES]:
   try:print('[OTHER POST FILE]',path.relative_to(ROOT),'COLUMNS',pq.ParquetFile(path).schema_arrow.names[:22])
   except Exception:print('[OTHER FILE]',path)
 print('[INFO] For a nonstandard post file, use --posts C:/.../your_posts.parquet')

def _normalize_text(text, rule):
 """Candidate interpretations ONLY. Exact per-text post-count parity is mandatory."""
 if pd.isna(text):return None
 s=str(text)
 if rule=='identity':return s
 if rule=='strip':return s.strip()
 if rule=='collapse':return ' '.join(s.split())
 if rule=='lower':return s.lower()
 if rule=='lower_strip':return s.lower().strip()
 if rule=='lower_collapse':return ' '.join(s.lower().split())
 if rule=='casefold_collapse':return ' '.join(s.casefold().split())
 if rule=='nfkc_collapse':return ' '.join(unicodedata.normalize('NFKC',s).split())
 if rule=='nfkc_lower_collapse':return ' '.join(unicodedata.normalize('NFKC',s).lower().split())
 if rule=='nfc_lower_collapse':return ' '.join(unicodedata.normalize('NFC',s).lower().split())
 if rule=='strip_invisible_collapse':
  return ' '.join(s.replace('\u200b','').replace('\ufeff','').split())
 raise ValueError(rule)

def _provenance_scan():
 """Locate the source that produced the exact-text family data; no source is executed."""
 import ast
 print('[PROVENANCE] Scanning project Python source for the original unique-text family producer',flush=True)
 candidates=[]
 for p in sorted((ROOT/'src').glob('*.py')):
  if p.name.startswith(('42_','43_','44_','45_','46_','47_','48_','49_')):continue
  try:source=p.read_text(encoding='utf-8-sig')
  except Exception:continue
  if 'unique_text_families.parquet' in source or ('normalized_text' in source and 'post_count' in source):
   lines=source.splitlines()
   key_lines=[i for i,l in enumerate(lines) if 'unique_text_families.parquet' in l or ('normalized_text' in l and any(k in l for k in ['=', 'rename','groupby','assign']))]
   print(f'[PRODUCER CANDIDATE] {p.relative_to(ROOT)}',flush=True)
   for i in key_lines[:12]:
    print(f'  {i+1}: {lines[i][:200]}',flush=True)
   candidates.append(str(p.relative_to(ROOT)))
 if not candidates:print('[PROVENANCE] No producer script discovered under src/. Check archived/ or notebooks.',flush=True)
 return candidates

def _unambiguous_index(keys, text_ids):
 """Construct lookup only where canonical key identifies one and only one mapping ID."""
 d=pd.DataFrame({'key':keys,'text_id':text_ids}).dropna(subset=['key'])
 d=d[d['key']!='']
 groups=d.groupby('key',sort=False)['text_id'].agg(['size','first'])
 unique=groups[groups['size']==1]
 return pd.Series(unique['first'].astype(np.int64).to_numpy(),index=unique.index)

def _canon(s, variant):
 if pd.isna(s):return None
 s=str(s)
 if variant in ('html_nfkc','html2_nfkc'):
  s=html.unescape(s)
  if variant=='html2_nfkc':s=html.unescape(s)
 elif variant=='nfkc_html':s=html.unescape(unicodedata.normalize('NFKC',s))
 if variant in ('strip_controls','strip_controls_html'):
  if variant=='strip_controls_html':s=html.unescape(s)
  s=re.sub(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u2060\u00ad\u202a-\u202e]', '', s)
 if variant=='literal_escapes':
  s=s.replace('\\r\\n',' ').replace('\\n',' ').replace('\\t',' ')
 if variant=='remove_urls':s=re.sub(r'https?://\S+', ' ', s)
 return ' '.join(unicodedata.normalize('NFKC',s).split())

def _producer_normalized_text(posts):
 """Evaluate ONLY the normalized_text assignment from the actual upstream Script 26.

 The script itself is never imported or executed; its main I/O and output-writing
 operations are excluded. This reuses the producer's own expression and helpers.
 """
 import ast
 source_file=ROOT/'src'/'26_lineage_foundation.py'
 if not source_file.exists():
  print('[PRODUCER] Source missing:',source_file,flush=True)
  return None
 source=source_file.read_text(encoding='utf-8-sig')
 tree=ast.parse(source,filename=str(source_file))
 def is_norm_target(target):
  return (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name)
          and target.value.id=='df' and
          isinstance(target.slice,ast.Constant) and target.slice.value=='normalized_text')
 assignments=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(is_norm_target(t) for t in node.targets):assignments.append(node)
  if isinstance(node,ast.AnnAssign) and is_norm_target(node.target):assignments.append(node)
 if not assignments:
  print('[PRODUCER] Could not locate df["normalized_text"] assignment in Script 26.',flush=True)
  return None
 assignments.sort(key=lambda x:x.lineno)
 node=assignments[-1]
 print(f'[PRODUCER] Found exact normalized_text expression in {source_file.name}, lines {node.lineno}-{node.end_lineno}',flush=True)
 # Only import definitions and simple constants, never execute the original
 # top-level pipeline, and never write to its output locations.
 globals_dict={'df':posts.copy(), 'pd':pd,'np':np,'re':re,'unicodedata':unicodedata,'html':html,
               '__name__':'_lineage_normalization_probe_'}
 helper_nodes=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom) and n.module=='__future__':continue
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.AsyncFunctionDef)):
   # Filter automatic main functions if present; definitions have no side effects.
   helper_nodes.append(n)
  elif isinstance(n,(ast.Assign,ast.AnnAssign)):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target]
   if all(isinstance(t,ast.Name) for t in targets):
    value=n.value
    # Only static constants, not data loading or output-writing calls.
    try:ast.literal_eval(value)
    except (ValueError,TypeError,MemoryError,RecursionError):continue
    helper_nodes.append(n)
 try:
  if helper_nodes:
   prelude=ast.fix_missing_locations(ast.Module(body=helper_nodes,type_ignores=[]))
   exec(compile(prelude,str(source_file),'exec'),globals_dict)
  expr=ast.Expression(body=node.value)
  result=eval(compile(ast.fix_missing_locations(expr),str(source_file),'eval'),globals_dict)
  if not isinstance(result,(pd.Series,np.ndarray,list,tuple)):
   raise TypeError(f'Expected per-row Series, got {type(result).__name__}')
  result=pd.Series(result,index=posts.index) if not isinstance(result,pd.Series) else result
  if len(result)!=len(posts):raise ValueError('producer expression returned a different number of post rows')
  print(f'[PRODUCER] Applied Script 26 expression to {len(result):,} source posts (no pipeline rerun).',flush=True)
  return result
 except Exception as e:
  print(f'[PRODUCER] Exact expression evaluation failed: {type(e).__name__}: {e}',flush=True)
  print('[PRODUCER] Please share lines 110-145 of src/26_lineage_foundation.py if this persists.',flush=True)
  return None

def _reconcile_ids(df,mapping,force_column=None,force_rule=None):
 """Lossless join. The exact original producer is authoritative; probes are diagnostics only.

 A candidate is accepted iff ALL post counts agree for EVERY text ID and no
 originally null post is silently introduced. Never fuzzy-match or force counts.
 """
 exp=mapping.sort_values('text_id').reset_index(drop=True)
 ids_int=exp.text_id.to_numpy(dtype=np.int64)
 if not np.array_equal(ids_int,np.arange(len(exp))):
  raise RuntimeError('text_id mapping is not contiguous 0..N-1')
 if exp.normalized_text.isna().any() or exp.normalized_text.duplicated().any():
  raise RuntimeError('Mapping normalized_text not unique/non-null')
 idx=pd.Index(exp.normalized_text.astype(str),name='normalized_text')
 # First use the upstream producer's ACTUAL normalizer, not guessed variants.
 # Compare exact counts per text_id before accepting any reconstruction.
 from_producer=_producer_normalized_text(df)
 if from_producer is not None:
  candidate=idx.get_indexer(from_producer.fillna('').astype(str).to_numpy())
  counts=exp.post_count.to_numpy(dtype=np.int64)
  found=int((candidate>=0).sum())
  observed=np.bincount(candidate[candidate>=0],minlength=len(counts))
  wrong=int((observed!=counts).sum())
  print(f'[JOIN PRODUCER] matched={found:,}/{int(counts.sum()):,}; mismatched text IDs={wrong:,}',flush=True)
  if found==int(counts.sum()) and wrong==0:
   print('[JOIN] VERIFIED with exact Script 26 normalization and per-text post counts.',flush=True)
   return candidate.astype(np.int64),'Message','producer_script26'
  # Save row-level *indices* and text IDs only, not sensitive message text.
  OUT.mkdir(parents=True,exist_ok=True)
  bad=np.flatnonzero(candidate<0)
  mismatch=np.flatnonzero(observed!=counts)
  (OUT/'producer_join_diagnostic.json').write_text(json.dumps({
   'producer_file':str(ROOT/'src'/'26_lineage_foundation.py'),
   'matched_posts':found,'expected_posts':int(counts.sum()),
   'unmatched_row_indices_first_30':bad[:30].tolist(),
   'mismatched_text_ids_first_30':mismatch[:30].tolist(),
   'mismatched_text_id_count':wrong
  },indent=2),encoding='utf-8')
  print('[PRODUCER] Exact normalizer does not reconcile the saved mapping; diagnostic file written.',flush=True)
 expected_counts=exp.post_count.to_numpy(dtype=np.int64)
 expected=int(expected_counts.sum())
 print(f'[RECONCILE] Expected {expected:,} post occurrences and {len(mapping):,} unique text IDs',flush=True)
 audit=[]
 best=('', '', -1, None)

 def check(ids,col,rule):
  nonlocal best
  ids=np.asarray(ids,dtype=np.int64)
  good=ids>=0; n=int(good.sum())
  counts=np.bincount(ids[good],minlength=len(expected_counts))
  mismatches=int(np.count_nonzero(counts!=expected_counts))
  audit.append((col,rule,n,mismatches))
  if n>best[2] or (n==best[2] and mismatches<(best[3] or 10**9)):best=(col,rule,n,mismatches)
  print(f'[JOIN CANDIDATE] {col}/{rule} matched={n:,} count_mismatched={mismatches:,}',flush=True)
  if n==expected and mismatches==0:
   print(f'[JOIN] VERIFIED total and every exact text-version count using {col}/{rule}',flush=True)
   return True
  return False

 cols=[]
 if force_column:
  cols=[force_column] if force_column in df else []
 else:
  cols=[c for c in ('text_id','unique_text_id','exact_family_id','normalized_text','Message','text_clean') if c in df]
 for col in cols:
  if col in ('text_id','unique_text_id'):
   v=pd.to_numeric(df[col],errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
   v[(v<0)|(v>=len(expected_counts))]=-1
   if check(v,col,'direct_id'):return v,col,'direct_id'
   continue
  if col=='exact_family_id':
   x=exp[['exact_family_id','text_id']].dropna().copy()
   x['exact_family_id']=x.exact_family_id.astype(str)
   lookup=_unambiguous_index(x.exact_family_id.to_numpy(dtype=str),x.text_id.to_numpy(dtype=np.int64))
   v=df[col].astype('string').map(lookup).fillna(-1).to_numpy(dtype=np.int64)
   if check(v,col,'exact_family_id'):return v,col,'exact_family_id'
   continue
  # Transform unique source strings once; keep original post order intact.
  source=df[col]
  uniques=source.dropna().astype(str).drop_duplicates()
  rules=([force_rule] if force_rule else ['identity','nfkc_collapse','html_nfkc','html2_nfkc','nfkc_html','strip_controls','strip_controls_html','literal_escapes'])
  base=None
  for rule in rules:
   if rule=='identity':values=uniques.to_numpy(dtype=object)
   elif rule=='nfkc_collapse':values=uniques.map(lambda t:' '.join(unicodedata.normalize('NFKC',t).split())).to_numpy(dtype=object)
   else:values=uniques.map(lambda t:_canon(t,rule)).to_numpy(dtype=object)
   key_to_id=pd.Series(idx.get_indexer(values),index=uniques.to_numpy(dtype=object))
   mapped=source.astype('string').map(key_to_id).fillna(-1).to_numpy(dtype=np.int64)
   if check(mapped,col,rule):return mapped,col,rule
   if rule=='nfkc_collapse':base=mapped.copy()
  if col=='Message' and base is not None:
   # Mixed fallback: only fill otherwise unmatched posts where the proposed
   # canonicalization identifies exactly ONE original mapping ID.
   merged=base.copy()
   source_unique=source.dropna().astype(str).drop_duplicates()
   for rule in ('html_nfkc','html2_nfkc','strip_controls','strip_controls_html','literal_escapes'):
    v=source_unique.map(lambda t:_canon(t,rule)).to_numpy(dtype=object)
    lookup_idx=idx.get_indexer(v)
    remap=pd.Series(lookup_idx,index=source_unique.to_numpy(dtype=object))
    new=source.astype('string').map(remap).fillna(-1).to_numpy(dtype=np.int64)
    # Disallow evidence contradictions between direct NFKC and fallbacks.
    conflict=(merged>=0)&(new>=0)&(merged!=new)
    fill=(merged<0)&(new>=0)
    print(f'[FALLBACK] {rule}: new_matches={int(fill.sum()):,}, conflicting={int(conflict.sum()):,}',flush=True)
    if not conflict.any():merged[fill]=new[fill]
   if check(merged,col,'nfkc_plus_nonconflicting_fallbacks'):return merged,col,'nfkc_plus_nonconflicting_fallbacks'

 OUT.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(audit,columns=['post_column','normalization','matched_posts','count_mismatched_text_ids']).to_csv(OUT/'post_join_diagnostics.csv',index=False)
 _provenance_scan()
 # Report only aggregates: no raw social-media post text leaves the user's machine.
 print(f'[JOIN] Best candidate: {best}; still requires {expected:,} posts and every per-text count matched.',flush=True)
 print('[JOIN] Diagnostics saved:',OUT/'post_join_diagnostics.csv',flush=True)
 raise RuntimeError('No verified lossless join. The exact upstream creator of unique_text_families.parquet must be used; no partial estimates generated.')

def _parse_timestamp_candidate(values, label):
 """Parse a saved timestamp column WITHOUT requiring a single date format.

 Return (timestamps, parse_fraction, has_subday_information, strategy).
 Dates are assumed UTC only when no timezone information was supplied;
 record that assumption in the output rather than silently converting a timezone.
 """
 from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
 series=pd.Series(values).reset_index(drop=True)
 def plausible(x):
  if x is None:return False
  n=x.notna().sum()
  if not n:return False
  y=x.dropna().dt.year
  # This dataset spans 2021--2024; tolerate some off-window records for QA.
  return float(y.between(2018,2030).mean())>=0.995
 def final(dt,method):
  if not plausible(dt):return None
  fine=bool(((((dt.dt.hour!=0)|(dt.dt.minute!=0)|(dt.dt.second!=0)) & dt.notna()).any()))
  return (dt,float(dt.notna().mean()),fine,method)
 if is_datetime64_any_dtype(series):
  dt=pd.to_datetime(series,utc=True,errors='coerce')
  v=final(dt,'datetime_dtype')
  if v:return v
 if is_numeric_dtype(series):
  n=pd.to_numeric(series,errors='coerce')
  for unit in ('s','ms','us','ns'):
   dt=pd.to_datetime(n,unit=unit,utc=True,errors='coerce')
   v=final(dt,'numeric_unix_'+unit)
   if v:return v
 # Probe a small, evenly spaced sample BEFORE parsing the entire column.
 # A malformed CrowdTangle field could otherwise invoke dateutil hundreds of
 # thousands of times, despite the preprocessed `date` field already existing.
 text=series.astype('string').str.strip()
 if len(text)==0:
  empty=pd.Series(pd.NaT,index=series.index,dtype='datetime64[ns, UTC]')
  return empty,0.0,False,'empty'
 sample=text.iloc[np.unique(np.linspace(0,len(text)-1,num=min(len(text),128),dtype=int))]
 best=None
 for name,opts in [
  ('mixed',{'format':'mixed'}),
  ('iso8601',{'format':'ISO8601'}),
  ('dayfirst_mixed',{'format':'mixed','dayfirst':True}),
 ]:
  try:
   with __import__('warnings').catch_warnings():
    __import__('warnings').simplefilter('ignore',UserWarning)
    probe=pd.to_datetime(sample,utc=True,errors='coerce',**opts)
   if float(probe.notna().mean())<0.70:
    continue
   dt=pd.to_datetime(text,utc=True,errors='coerce',**opts)
  except (ValueError,TypeError):continue
  v=final(dt,name)
  if v and v[1]>=0.98:return v
  if v and (best is None or v[1]>best[1]):best=v
 # Only strip explicit UTC/GMT/Z suffixes; unknown timezone codes are not
 # silently discarded or mistaken for UTC.
 stripped_sample=sample.str.replace(r'(?i)\s+(?:UTC|GMT|Z)$','',regex=True)
 if not stripped_sample.equals(sample):
  try:
   probe=pd.to_datetime(stripped_sample,format='mixed',utc=True,errors='coerce')
   if float(probe.notna().mean())>=.70:
    stripped=text.str.replace(r'(?i)\s+(?:UTC|GMT|Z)$','',regex=True)
    dt=pd.to_datetime(stripped,format='mixed',utc=True,errors='coerce')
    v=final(dt,'removed_explicit_UTC_suffix')
    if v and v[1]>=0.98:return v
    if v and (best is None or v[1]>best[1]):best=v
  except (ValueError,TypeError):pass
 if best is not None:return best
 empty=pd.Series(pd.NaT,index=series.index,dtype='datetime64[ns, UTC]')
 return empty,0.0,False,'unparsed'


def _resolve_post_timestamps(df):
 """Select best reliable existing timestamp; do not assume Post Created parses.

 All viable alternatives (including preprocessing's `date`) are assessed.
 The chosen column, granularity and coverage are surfaced for reproducibility.
 """
 names=[c for c in ALIASES['date'] if c in df.columns]
 for c in df.columns:
  if c not in names and ('timestamp' in str(c).lower() or str(c).lower() in ('datetime','created')):
   names.append(c)
 candidates=[]
 if 'Post Created Date' in df and 'Post Created Time' in df:
  combined=df['Post Created Date'].astype('string').fillna('')+' '+df['Post Created Time'].astype('string').fillna('')
  candidates.append(('Post Created Date + Post Created Time',combined))
 for n in names:candidates.append((n,df[n]))
 successful=[]
 for label,values in candidates:
  dt,coverage,subday,method=_parse_timestamp_candidate(values,label)
  print(f'[DATE CANDIDATE] {label}: parsed={coverage:.2%}; clock_times={subday}; parser={method}',flush=True)
  if coverage>=.98:
   successful.append((label,dt,coverage,subday,method))
 if not successful:
  # These are only timestamp values, not message text or personal content.
  for label,values in candidates[:3]:
   sample=values.dropna().astype(str).head(3).tolist()
   print(f'[DATE EXAMPLE] {label} type={values.dtype}: {sample}',flush=True)
  raise ValueError('None of the available date columns parsed at >=98%. See [DATE EXAMPLE]. '
                   'No downstream estimates were produced.')
 # Prefer full timestamps over date-only data, then better coverage, then
 # original source order. This avoids midnight-only `date` trumping detailed time.
 successful.sort(key=lambda x:(x[3],x[2]),reverse=True)
 label,dt,coverage,subday,method=successful[0]
 if len(successful)>1:
  # Calendar dates may differ slightly because of timezone offsets. Detect
  # gross differences; don't silently choose incompatible timelines.
  for other_label,other_dt,other_cov,other_subday,_ in successful[1:]:
   overlap=dt.notna() & other_dt.notna()
   if overlap.sum()==0:continue
   different=((dt[overlap]-other_dt[overlap]).abs()>pd.Timedelta(days=2)).mean()
   if different>.005:
    raise ValueError(f'Timestamp columns {label} and {other_label} disagree by >2 days '
                     f'on {different:.2%} of overlapping posts. Stop: inspect the source date fields.')
 print(f'[DATE] Selected {label} ({method}); parsed={coverage:.2%}; '
       f'resolution={"subday" if subday else "date-only"}; '
       f'range={dt.min()} to {dt.max()}',flush=True)
 if not subday:
  print('[WARNING] Only day-level dates are available. Sub-day circulation timing '
        'cannot be reproduced; interpret temporal and lifespan comparisons accordingly.',flush=True)
 return dt,label,method,subday


def load_posts(path,mapping,join_only=False):
 if not path.exists():raise FileNotFoundError(f'Post-level file not found: {path}. Run --inspect and pass --posts.')
 import pyarrow.parquet as pq
 available=pq.ParquetFile(path).schema_arrow.names
 candidates={'Message','text_clean','normalized_text','cleaned_text','message_normalized',
             'text_id','unique_text_id','exact_family_id'}
 # Load ALL candidate date fields, not just the first alias. In these records,
 # Post Created may be unparseable even when the precomputed `date` is sound.
 candidates.update(c for c in ALIASES['date'] if c in available)
 if 'Post Created Time' in available:candidates.add('Post Created Time')
 for key in ('page','category','country'):
  col=alias(available,key)
  if col:candidates.add(col)
 columns=[c for c in available if c in candidates]
 df=pd.read_parquet(path,columns=columns)
 print(f'[POSTS] {path} records={len(df):,} selected_columns={columns}',flush=True)
 ids,col,rule=_reconcile_ids(df,mapping)
 found=ids>=0
 df=df.loc[found].copy()
 df['text_id']=ids[found].astype(np.int64)
 assert len(df)==int(mapping.post_count.sum()), 'Post-count parity lost unexpectedly'
 if join_only:
  print(f'[DONE] Joined {len(df):,} posts safely via {col}/{rule}; no downstream calculations requested.',flush=True)
  return None
 page=alias(df.columns,'page')
 if not page:raise ValueError(f'Post file requires page column. Actual: {list(df.columns)}')
 dt,t,method,subday=_resolve_post_timestamps(df)
 # Dates are returned as a new default RangeIndex, so align by position.
 df['_date']=dt.to_numpy()
 df['_page']=df[page].astype('string').fillna('UNKNOWN').astype(str)
 for target,kind in [('_category','category'),('_country','country')]:
  c=alias(df.columns,kind)
  df[target]=df[c].astype('string').fillna('UNKNOWN').astype(str) if c else 'UNKNOWN'
 print(f'[JOIN] exact post-count parity verified: {len(df):,}; time={t}; page={page}',flush=True)
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'date_provenance.json').write_text(json.dumps({
  'selected_source':t,'parser':method,'has_subday_resolution':subday,
  'parsable_fraction':float(dt.notna().mean()),'original_post_records':len(df)
 },indent=2),encoding='utf-8')
 return df[['text_id','_date','_page','_category','_country']]

def read_members(name,mapping):
 p=MEMBERS[name]
 if not p.exists():raise FileNotFoundError(f'Missing repaired member file for {name}: {p}')
 d=pd.read_parquet(p,columns=['text_id','repaired_lineage_id'])
 if d.text_id.duplicated().any():raise RuntimeError(f'Duplicate text IDs in {p}')
 if not set(d.text_id).issubset(set(mapping.text_id)):raise RuntimeError(f'{name} contains unknown text IDs')
 return d

def summarize(name,posts,mapping,mem,emb=None):
 # Multi-version families, plus recurrent exact-only families (post count >=2).
 mem=mem.rename(columns={'repaired_lineage_id':'lineage'}).copy()
 mem['lineage']=name+'_M_'+mem.lineage.astype(str)
 exact=mapping.loc[(mapping.post_count>=2)&(~mapping.text_id.isin(mem.text_id)),['text_id']].copy()
 exact['lineage']=name+'_E_'+exact.text_id.astype(str)
 all_mem=pd.concat([mem,exact],ignore_index=True)
 p=posts.merge(all_mem,on='text_id',how='inner',validate='many_to_one')
 if p.empty:raise RuntimeError(f'No linked posts for {name}')
 per=p.groupby('lineage',sort=False)
 family=per.agg(posts=('text_id','size'),versions=('text_id','nunique'),pages=('_page','nunique'),
                categories=('_category','nunique'),countries=('_country','nunique'),
                first=('_date','min'),last=('_date','max')).reset_index()
 family['lifespan_days']=(family['last']-family['first']).dt.total_seconds()/86400
 # Earliest available first appearance per unique text version.
 version=p.groupby(['lineage','text_id'],sort=False)['_date'].min().reset_index(name='first_seen')
 page_first=p.groupby(['lineage','_page'],sort=False)['_date'].min().reset_index(name='first_seen')
 v2=version.sort_values(['lineage','first_seen','text_id']).groupby('lineage',sort=False).nth(1).reset_index()[['lineage','first_seen']].rename(columns={'first_seen':'second_version_at'})
 p2=page_first.sort_values(['lineage','first_seen','_page']).groupby('lineage',sort=False).nth(1).reset_index()[['lineage','first_seen']].rename(columns={'first_seen':'second_page_at'})
 family=family.merge(v2,on='lineage',how='left').merge(p2,on='lineage',how='left')
 family['days_to_second_page']=(family.second_page_at-family['first']).dt.total_seconds()/86400
 family['days_to_second_version']=(family.second_version_at-family['first']).dt.total_seconds()/86400
 # Country is page-admin metadata only.
 fv=family[family.versions>=2]
 out={
  'variant':name,'total_input_posts':len(posts),'recurrent_families':len(family),
  'multi_version_families':len(fv),'exact_only_families':len(family)-len(fv),
  'recurrent_posts':len(p),'recurrent_post_coverage':len(p)/len(posts),
  'cross_page_families':int((family.pages>=2).sum()),
  'cross_page_family_share':float((family.pages>=2).mean()),
  'cross_country_families':int((family.countries>=2).sum()),
  'median_lifespan_days':float(family.lifespan_days.median()),
  'median_days_to_second_page':float(family.days_to_second_page.median()),
  'median_days_to_second_version':float(family.days_to_second_version.median()),
  'multi_version_cross_page_share':float((fv.pages>=2).mean()),
  'multi_version_cross_category_share':float((fv.categories>=2).mean()),
 }
 OUT.mkdir(parents=True,exist_ok=True)
 family.drop(columns=['first','last','second_page_at','second_version_at']).to_csv(OUT/f'{name}_family_summary.csv',index=False)
 if emb is not None:
  temp=version.merge(fv[['lineage']],on='lineage',how='inner')
  temp=temp.sort_values(['lineage','first_seen','text_id']).reset_index(drop=True)
  temp['first_id']=temp.groupby('lineage').text_id.transform('first')
  temp['first_date']=temp.groupby('lineage').first_seen.transform('first')
  temp=temp[temp.text_id!=temp.first_id].copy()
  temp['t_days']=(temp.first_seen-temp.first_date).dt.total_seconds()/86400
  id1=temp.first_id.to_numpy(dtype=np.int64);id2=temp.text_id.to_numpy(dtype=np.int64)
  if max(id1.max(),id2.max())>=len(emb):raise RuntimeError('E5 cache smaller than observed text IDs')
  E1=np.asarray(emb[id1],dtype=np.float32);E2=np.asarray(emb[id2],dtype=np.float32)
  norm=(np.linalg.norm(E1,axis=1)*np.linalg.norm(E2,axis=1))
  temp['distance']=1-(E1*E2).sum(axis=1)/np.maximum(norm,1e-12)
  temp['log_elapsed']=np.log1p(temp.t_days.clip(lower=0))
  temp['family_size']=temp.groupby('lineage').text_id.transform('size')
  # Fixed effects: within-family demeaning, cluster-robust inference.
  temp['x']=temp.log_elapsed-temp.groupby('lineage').log_elapsed.transform('mean')
  temp['y']=temp.distance-temp.groupby('lineage').distance.transform('mean')
  use=temp.groupby('lineage').x.transform(lambda s:float((s*s).sum())>1e-12)
  fitdata=temp[use].copy()
  import statsmodels.api as sm
  if len(fitdata)>2 and fitdata.lineage.nunique()>1:
   fit=sm.OLS(fitdata.y,fitdata[['x']]).fit(cov_type='cluster',cov_kwds={'groups':fitdata.lineage})
   beta=float(fit.params['x']);se=float(fit.bse['x']);CI=fit.conf_int().loc['x'].tolist()
   out.update({'temporal_beta':beta,'temporal_se':se,'temporal_ci_low':float(CI[0]),
               'temporal_ci_high':float(CI[1]),'temporal_n_versions':len(fitdata),
               'temporal_n_families':fitdata.lineage.nunique(),
               'mean_temporal_distance':float(fitdata.distance.mean()),
               'sd_temporal_distance':float(fitdata.distance.std())})
   # Same family-weight total contribution: observations weight inversely to family sizes.
   wm=fitdata.groupby('lineage').x.transform('size').rdiv(1.0)
   wfit=sm.WLS(fitdata.y,fitdata[['x']],weights=wm).fit(cov_type='cluster',cov_kwds={'groups':fitdata.lineage})
   out['temporal_family_weighted_beta']=float(wfit.params['x'])
  temp[['lineage','text_id','first_id','t_days','distance','log_elapsed']].to_parquet(OUT/f'{name}_version_distances.parquet',index=False)
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--posts',help='Post-level file with text ID or normalized text, timestamp and page metadata')
 ap.add_argument('--inspect',action='store_true')
 ap.add_argument('--skip-temporal',action='store_true')
 ap.add_argument('--join-only',action='store_true',help='Check post-text mapping without running downstream models')
 a=ap.parse_args()
 if a.inspect:inspect();return
 if not MAPPING.exists():raise FileNotFoundError(f'Cannot find original mapping: {MAPPING}')
 mapping=pd.read_parquet(MAPPING).sort_values('text_id').reset_index(drop=True)
 needed={'text_id','exact_family_id','normalized_text','post_count'}
 if not needed.issubset(mapping.columns):raise ValueError(f'Missing mapping fields {needed-set(mapping.columns)}')
 if mapping.text_id.duplicated().any():raise ValueError('Duplicate text IDs')
 if a.posts:pp=Path(a.posts).expanduser();pp=pp if pp.is_absolute() else ROOT/pp
 else:
  found=[ROOT/c for c in POST_CANDIDATES if (ROOT/c).exists()]
  if len(found)!=1:raise RuntimeError(f'Found {len(found)} candidate post files; run --inspect and specify --posts explicitly. Candidates: {found}')
  pp=found[0]
 posts=load_posts(pp,mapping,join_only=a.join_only)
 if a.join_only:return
 emb=None
 if not a.skip_temporal:
  if EMBED.exists():
   emb=np.load(EMBED,mmap_mode='r')
   if emb.shape[0]!=len(mapping):raise ValueError(f'E5 embeddings rows={emb.shape[0]} but mapping rows={len(mapping)}')
  else:print('[WARNING] Missing E5 embeddings; temporal regressions skipped',flush=True)
 rows=[]
 for k in MEMBERS:
  print('[ANALYZE]',k,flush=True)
  r=summarize(k,posts,mapping,read_members(k,mapping),emb)
  rows.append(r)
  if k=='baseline' and (r['recurrent_families']!=21441 or r['recurrent_posts']!=81629):
   print(f'[IMPORTANT] Baseline inventory does not reproduce paper (21,441 families / 81,629 posts): {r["recurrent_families"]:,} families / {r["recurrent_posts"]:,} posts. Verify original corpus and membership before using revised values.',flush=True)
  print('[RESULT]',json.dumps(r,default=str),flush=True)
 OUT.mkdir(parents=True,exist_ok=True)
 df=pd.DataFrame(rows)
 df.to_csv(OUT/'downstream_comparison.csv',index=False)
 (OUT/'downstream_comparison.json').write_text(df.to_json(orient='records',indent=2),encoding='utf-8')
 if 'temporal_beta' in df:
  baseline=float(df.set_index('variant').loc['baseline','temporal_beta'])
  if not np.isclose(baseline,.001497,atol=.00020):
   print(f'[IMPORTANT] Baseline beta {baseline:.6f} does NOT closely reproduce published .001497. Investigate original temporal filtering/date logic before using cleaned beta in paper.',flush=True)
  else:print('[CHECK] Baseline beta approximately reproduces original paper',flush=True)
 print('[DONE]',OUT/'downstream_comparison.csv',flush=True)
if __name__=='__main__':main()
