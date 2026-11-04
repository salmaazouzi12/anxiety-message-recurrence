#!/usr/bin/env python3
"""Stage 4: reproducible ablations of the REAL anxiety reconstruction (Scripts 34/35).

Run from the root of anxiety-research:
  python src/38_stage4_faithful_ablation.py --mode cohesion
  python src/38_stage4_faithful_ablation.py --mode all

This script does NOT overwrite any output of Scripts 34/35. It generates
isolated copies whose only changes are explicit parameters / paths.

Modes:
- cohesion (default): evaluate saved baseline + rerun original Script 35 with
  stricter flagged-family criteria and/or complete linkage.
- all: same plus strip URLs/hashtags/frequent generic footers, re-score all
  available raw MinHash and E5-novel candidates on cleaned text using the EXACT
  HashingVectorizer settings of Scripts 28/33, then rerun real Scripts 34/35.
  Retrieval candidates themselves remain fixed; this is a fixed-pool ablation,
  not re-running MinHash/FAISS. The lexical floor is fixed >=0.86.

Outputs in outputs/anxiety_lineages/stage4_full_ablation/.
NOTE: Validation is conditional on a 400-pair STRATIFIED sample, and includes
53 AI-proposed labels reviewed/accepted by the researcher. Not population recall.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name.lower() == 'src' else Path(__file__).resolve().parent
SRC = ROOT / 'src'
BASE = ROOT / 'outputs' / 'anxiety_lineages'
OUT = BASE / 'stage4_full_ablation'
GENERATED = SRC / 'stage4_generated'

# Hard fail if the source scripts differ from inspected structural markers.
REQUIRED_34 = [
    'MINHASH = BASE / "minhash"',
    'LEXICAL = BASE / "lexical_scoring"',
    'SEMANTIC = BASE / "combined_retrieval" / "novel_e5_scored"',
    'OUT = BASE / "combined_lineages"',
    'CORE = 0.90', 'FLOOR = 0.86',
    'MINHASH / "text_id_mapping.parquet"',
]
REQUIRED_35 = [
    'OUT = BASE / "combined_cohesion_repair"',
    'THRESHOLD = 0.86',
    'MAX_LOW_SIM_FRACTION = 0.10',
    'hierarchical_repair(sim, "average")',
    'BASE / "combined_lineages"',
]


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch_once(text, old, new):
    need(text.count(old) == 1, f'Unsafe script patch: expected exactly one occurrence of {old!r}, found {text.count(old)}')
    return text.replace(old, new, 1)


def read_orig(which):
    path = SRC / which
    need(path.exists(), f'Missing original script: {path}')
    src = path.read_text(encoding='utf-8-sig')
    required = REQUIRED_34 if which.startswith('34_') else REQUIRED_35
    for marker in required:
        need(marker in src, f'{which} differs from the version provided. Missing marker: {marker!r}. No patch attempted.')
    return src


def generate_repair(name, threshold=0.86, max_low=0.10, first_linkage='average', source=None):
    src = read_orig('35_repair_combined_lineages.py')
    src = patch_once(src, 'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]')
    if source:
        original = 'SOURCE = (\n    BASE / "combined_lineages"\n    / "combined_lineage_members.parquet"\n)'
        new = f'SOURCE = (BASE / "stage4_full_ablation" / "{source}" / "combined_lineages" / "combined_lineage_members.parquet")'
        src = patch_once(src, original, new)
    src = patch_once(src, 'OUT = BASE / "combined_cohesion_repair"', f'OUT = BASE / "stage4_full_ablation" / "{name}" / "combined_cohesion_repair"')
    if threshold != 0.86:
        src = patch_once(src, 'THRESHOLD = 0.86', f'THRESHOLD = {threshold!r}')
    if max_low != 0.10:
        src = patch_once(src, 'MAX_LOW_SIM_FRACTION = 0.10', f'MAX_LOW_SIM_FRACTION = {max_low!r}')
    if first_linkage == 'complete':
        src = patch_once(src, 'labels = hierarchical_repair(sim, "average")', 'labels = hierarchical_repair(sim, "complete")')
        src = patch_once(src, 'method_used = "average"', 'method_used = "complete"')
    path = GENERATED / f'35_{name}.py'
    path.write_text(src, encoding='utf-8')
    return path


def generate_build_clean(name='deboilerplated'):
    src = read_orig('34_build_combined_lineages.py')
    src = patch_once(src, 'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[2]')
    src = patch_once(src, 'LEXICAL = BASE / "lexical_scoring"', f'LEXICAL = BASE / "stage4_full_ablation" / "{name}" / "lexical_scoring"')
    src = patch_once(src, 'SEMANTIC = BASE / "combined_retrieval" / "novel_e5_scored"', f'SEMANTIC = BASE / "stage4_full_ablation" / "{name}" / "combined_retrieval" / "novel_e5_scored"')
    src = patch_once(src, 'OUT = BASE / "combined_lineages"', f'OUT = BASE / "stage4_full_ablation" / "{name}" / "combined_lineages"')
    src = patch_once(src, 'MINHASH / "text_id_mapping.parquet"', f'BASE / "stage4_full_ablation" / "{name}" / "text_id_mapping_clean.parquet"')
    path = GENERATED / f'34_{name}.py'
    path.write_text(src, encoding='utf-8')
    return path


def run_original_variant(script, verbose=True):
    print(f'[EXEC] Exact source logic, isolated variant: {script.name}', flush=True)
    if verbose:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    else:
        with (OUT / 'run_logs.txt').open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)


def read_labels():
    options = [SRC, ROOT, OUT]
    def locate(filename):
        for folder in options:
            p = folder / filename
            if p.exists(): return p
        raise FileNotFoundError(f'Missing {filename}. Put it beside this script in src/.')
    labpath = locate('pair_annotations_completed.csv')
    keypath = locate('pair_sampling_key.csv')
    with labpath.open(encoding='utf-8-sig', newline='') as f: labels=list(csv.DictReader(f))
    with keypath.open(encoding='utf-8-sig', newline='') as f: keys=list(csv.DictReader(f))
    lookup = {r['pair_id']:r for r in labels}
    need(len(lookup)==400 and len(keys)==400, f'Expected 400 annotated pairs, got labels={len(lookup)} key={len(keys)}')
    need({r['pair_id'] for r in keys} == set(lookup), 'Annotation IDs and sampling key IDs do not match')
    items=[]
    for k in keys:
        v=lookup[k['pair_id']]
        gold=v['label'].strip().upper()
        need(gold in {'SAME','RELATED','UNRELATED'}, f'Unknown label {gold!r} for {k["pair_id"]}')
        items.append({'pair_id':k['pair_id'], 'a':int(k['text_id_a']), 'b':int(k['text_id_b']), 'gold':gold,
                      'source_group':k.get('source_group',''), 'sampling_stratum':k.get('sampling_stratum',''),
                      'annotation_origin':v.get('annotation_origin',''),
                      'original_sampling_decision':int(k.get('source_group','')=='same_reconstructed_family')})
    return items


def component_map(path):
    import pyarrow.parquet as pq
    need(path.exists(), f'Missing membership output {path}')
    schema = pq.ParquetFile(path).schema_arrow.names
    need('text_id' in schema and 'repaired_lineage_id' in schema, f'Unexpected membership schema {schema}')
    out={}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000, columns=['text_id','repaired_lineage_id']):
        d=batch.to_pydict()
        for tid,lid in zip(d['text_id'],d['repaired_lineage_id']):
            if tid in out: raise RuntimeError('Duplicate text_id in '+str(path))
            out[int(tid)]=int(lid)
    return out


def evaluate(name, labels, member_path):
    memberships=component_map(member_path)
    p=member_path.parent
    inv=p/'repaired_lineage_inventory.parquet'
    import pyarrow.parquet as pq
    row_cnt=pq.read_metadata(inv).num_rows if inv.exists() else len(set(memberships.values()))
    cases=[]
    counts=Counter()
    for row in labels:
        a,b=row['a'],row['b']
        pred=int(a in memberships and b in memberships and memberships[a] == memberships[b])
        truth=int(row['gold']=='SAME')
        outcome=('TP' if pred and truth else 'FP' if pred else 'FN' if truth else 'TN')
        counts[outcome]+=1
        cases.append({'variant':name,**row,'pred_same':pred,'reference_same':truth,'outcome':outcome,
                      'matches_sampling_key_decision':int(pred==row['original_sampling_decision'])})
    tp,fp,fn,tn=[counts[x] for x in ['TP','FP','FN','TN']]
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    extra={}
    summary_json=p/'cohesion_repair_summary.json'
    if summary_json.exists():
        extra=json.loads(summary_json.read_text(encoding='utf-8'))
    summary={'variant':name,'sample_n':len(labels),'TP':tp,'FP':fp,'FN':fn,'TN':tn,
             'sample_precision':precision,'sample_recall':recall,'sample_f1':f1,
             'recurrent_families':row_cnt,'recurrent_unique_text_ids':len(memberships),
             'posts_in_lineages':extra.get('posts_in_lineages'),
             'excluded_singletons':extra.get('excluded_singletons'),
             'membership_file':str(member_path.relative_to(ROOT)),
             'agreement_with_sampling_decision':sum(r['matches_sampling_key_decision'] for r in cases)/len(cases)}
    return summary,cases


def save_rows(path, rows):
    if not rows: return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction='ignore');w.writeheader();w.writerows(rows)


def clean_mapping(name):
    import pandas as pd
    source=BASE/'minhash'/'text_id_mapping.parquet'
    need(source.exists(),f'Missing {source}')
    m=pd.read_parquet(source)
    need({'text_id','normalized_text','post_count'}.issubset(m.columns),f'Mapping has columns {list(m.columns)}')
    m=m.sort_values('text_id').reset_index(drop=True)
    need((m['text_id'].to_numpy()==__import__('numpy').arange(len(m))).all(), 'text_id IDs are not contiguous: original Script 34 would also reject them')
    text=m['normalized_text'].fillna('').astype(str)
    url_re=re.compile(r'(?i)(?:https?://|www\.)[^\s]+')
    hashtag_re=re.compile(r'(?<![\w])#[\w\u0600-\u06ff]+', flags=re.UNICODE)
    # Only repetitive footer-style lines are removed, not arbitrary repeated prose.
    footer_re=re.compile(r'(?i)(?:follow (?:us|me)|subscribe|visit (?:our|my)|click (?:here|the link)|for more (?:info|information|updates)|all rights reserved|disclaimer|contact us)')
    tails=Counter()
    for value in text:
        lines=value.splitlines()
        if len(lines)<2:continue
        tail=lines[-1].strip()
        if 15<=len(tail)<=180 and footer_re.search(tail): tails[tail.lower()]+=1
    redundant={s for s,c in tails.items() if c>=50}
    stats=Counter()
    def clean(v):
        old=v
        v=url_re.sub(' ',v)
        v=hashtag_re.sub(' ',v)
        lines=v.splitlines()
        if len(lines)>1 and lines[-1].strip().lower() in redundant:
            lines=lines[:-1];stats['repeat_footer_removed']+=1
        v=' '.join(' '.join(lines).split())
        if v!=old:stats['texts_modified']+=1
        if not v:stats['empty_after_cleaning']+=1
        return v
    m['normalized_text']=[clean(v) for v in text]
    out=OUT/name/'text_id_mapping_clean.parquet'
    out.parent.mkdir(parents=True,exist_ok=True)
    m.to_parquet(out,index=False)
    report={'texts':len(m),'footer_types_removed':len(redundant), **stats,
            'description':'URLs, full hashtag tokens, and high-frequency footer lines only; candidate retrieval is held fixed.'}
    (OUT/name/'text_cleaning_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('[CLEAN] '+json.dumps(report),flush=True)
    return m['normalized_text'].fillna('').astype(str).tolist()


def rescore_candidates(name, texts):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    from sklearn.feature_extraction.text import HashingVectorizer
    v=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**18,alternate_sign=False,norm='l2',dtype=np.float32)
    origins=[('lexical_scoring',sorted((BASE/'minhash').glob('candidate_pairs_*.parquet')),
              OUT/name/'lexical_scoring','scored_pairs'),
             ('novel_e5_scored',[BASE/'combined_retrieval'/'e5_novel_pairs.parquet'],
              OUT/name/'combined_retrieval'/'novel_e5_scored','scored_semantic')]
    needed={'text_id_1','text_id_2'}
    for source_name,paths,dest,prefix in origins:
        need(paths and all(p.exists() for p in paths),f'Missing unfiltered candidate inputs for {source_name}: {paths[:3]}. Cannot run genuine de-boilerplate re-scoring from thresholded scored edges.')
        dest.mkdir(parents=True, exist_ok=True)
        for j,path in enumerate(paths,1):
            pf=pq.ParquetFile(path)
            need(needed.issubset(set(pf.schema_arrow.names)), f'Candidate file lacks IDs: {path}, {pf.schema_arrow.names}')
            target=dest/f'{prefix}_{j:05d}.parquet'
            # Ensure each run is reproducible: replace this experiment's own output.
            if target.exists():target.unlink()
            writer=None
            examined=retained=0
            try:
                for batch in pf.iter_batches(batch_size=10000,columns=['text_id_1','text_id_2']):
                    cols=batch.to_pydict()
                    a=np.asarray(cols['text_id_1'],dtype=np.int64)
                    b=np.asarray(cols['text_id_2'],dtype=np.int64)
                    if len(a)==0:continue
                    need((a>=0).all() and (b>=0).all() and (a<len(texts)).all() and (b<len(texts)).all(),f'Out of range candidate IDs: {path}')
                    examined+=len(a)
                    ids,inverse=np.unique(np.concatenate([a,b]),return_inverse=True)
                    X=v.transform([texts[int(i)] for i in ids])
                    k=len(a)
                    cosine=np.asarray(X[inverse[:k]].multiply(X[inverse[k:]]).sum(axis=1)).ravel()
                    keep=cosine >= 0.86
                    retained+=int(keep.sum())
                    if not keep.any():continue
                    result=pa.table({'text_id_1':pa.array(a[keep]), 'text_id_2':pa.array(b[keep]),
                                     'lexical_cosine':pa.array(cosine[keep].astype(np.float32)),
                                     'edge_type':pa.array(np.where(cosine[keep]>=0.90,'core','weak'))})
                    if writer is None: writer=pq.ParquetWriter(target,result.schema,compression='snappy')
                    writer.write_table(result)
            finally:
                if writer is not None: writer.close()
            print(f'[SCORE] {source_name} {j}/{len(paths)}: {examined:,} candidates, {retained:,} >=0.86',flush=True)
        need(any(dest.glob('*.parquet')),f'No scored edges survived cleaning in {dest}. Do not run reconstruction with empty candidate pool.')


def write_report(summaries):
    s={r['variant']:r for r in summaries}
    report=['# Stage 4 — faithful graph-cohesion and fixed-pool de-boilerplate sensitivity', '',
            'Source: original `34_build_combined_lineages.py` and `35_repair_combined_lineages.py`,',
            'copied with only parameter/path changes. Source files are never overwritten.', '',
            '| Variant | TP | FP | FN | TN | Sample precision | Sample recall | Families | Linked unique versions | Linked posts |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in summaries:
        def rate(v):return f'{v:.3f}' if v is not None else '—'
        report.append(f'| {x["variant"]} | {x["TP"]} | {x["FP"]} | {x["FN"]} | {x["TN"]} | {rate(x["sample_precision"])} | {rate(x["sample_recall"])} | {x["recurrent_families"]:,} | {x["recurrent_unique_text_ids"]:,} | {x["posts_in_lineages"] if x["posts_in_lineages"] is not None else "—"} |')
    report.extend(['','## Interpretation constraints',
        '- 400 pairs were deliberately stratified. Precision and recall are sample-conditional; no population-wide accuracy claims.',
        '- The human reference contains 53 AI-proposed labels accepted by the researcher. Adjudication and a held-out sample remain desirable.',
        '- `SAME` is the positive class. `RELATED` and `UNRELATED` are negative for strict textual reuse.',
        '- The baseline comes from the **saved combined repaired families** (not lexical-only repair).',
        '- Compare baseline predictions to `source_group` original key decisions in `stage4_pair_predictions.csv`; mismatches can indicate a snapshot/provenance difference.',
        '- Graph/cohesion variants rerun the actual Script 35 implementation on the unchanged combined preliminary families.',
        '- De-boilerplate variant re-scores the *original MinHash and novel E5 candidate pool*, using the exact HashingVectorizer configuration, then reruns Scripts 34 and 35. Candidate retrieval itself is not rerun and cannot capture newly retrievable candidates.',
        '- Do not equate a reduced family count with higher quality; compare false merges and missed links.',
        '- For full-corpus recurrence coverage in **posts**, use post counts from Script 35 summaries, not unique-text counts.',
        '- Any author-selected setting chosen by looking at these labels needs held-out validation.',
        ])
    (OUT/'stage4_comparison_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cohesion','all'],default='cohesion',help='all adds fixed-candidate-pool de-boilerplating and full rebuild (34+35)')
    parser.add_argument('--quiet',action='store_true',help='send the long original-script logs to stage4_full_ablation/run_logs.txt')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    GENERATED.mkdir(parents=True,exist_ok=True)
    labels=read_labels()
    orig34=read_orig('34_build_combined_lineages.py')
    orig35=read_orig('35_repair_combined_lineages.py')
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),
              'mode':args.mode,
              'source_34_sha256':hashlib.sha256(orig34.encode()).hexdigest(),
              'source_35_sha256':hashlib.sha256(orig35.encode()).hexdigest(),
              'annotation_source_n':len(labels),
              'model':'original source algorithms; isolated output folders'}
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    baseline=BASE/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
    preliminary=BASE/'combined_lineages'/'combined_lineage_members.parquet'
    need(baseline.exists(), f'Missing baseline {baseline}. Do not use lexical-only family file as substitution.')
    need(preliminary.exists(), f'Missing preliminary {preliminary}')
    configurations=[('baseline_saved',None),
        ('strict_low05',{'max_low':0.05}),
        ('strict_threshold088',{'threshold':0.88}),
        ('complete_first',{'first_linkage':'complete'})]
    summaries,cases=[],[]
    for name,kwargs in configurations:
        if kwargs is None:
            member=baseline
            print('[BASELINE] Using saved original combined-cohesion repaired lineages',flush=True)
        else:
            script=generate_repair(name,**kwargs)
            run_original_variant(script,verbose=not args.quiet)
            member=OUT/name/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
        summary,rows=evaluate(name,labels,member)
        summaries.append(summary); cases.extend(rows)
        print(f'[EVAL] {name}: TP={summary["TP"]} FP={summary["FP"]} FN={summary["FN"]} TN={summary["TN"]}; sampling-key concordance={summary["agreement_with_sampling_decision"]:.1%}',flush=True)
        if name=='baseline_saved' and summary['agreement_with_sampling_decision'] < 0.99:
            print('[WARNING] The saved repaired components do not reproduce all original sampling-key decisions.\n'
                  '          This may reflect snapshot/membership differences. Do not report as original algorithm accuracy until lineage provenance is reconciled.',flush=True)
        save_rows(OUT/'stage4_variant_summary.csv',summaries)
        save_rows(OUT/'stage4_pair_predictions.csv',cases)
        write_report(summaries)
    if args.mode=='all':
        print('[FULL] Preparing text-cleaned mapping',flush=True)
        cleaned=clean_mapping('deboilerplated')
        rescore_candidates('deboilerplated',cleaned)
        build=generate_build_clean('deboilerplated')
        run_original_variant(build,verbose=not args.quiet)
        repair=generate_repair('deboilerplated',source='deboilerplated')
        run_original_variant(repair,verbose=not args.quiet)
        member=OUT/'deboilerplated'/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
        summary,rows=evaluate('deboilerplated',labels,member)
        summaries.append(summary);cases.extend(rows)
        save_rows(OUT/'stage4_variant_summary.csv',summaries)
        save_rows(OUT/'stage4_pair_predictions.csv',cases)
        write_report(summaries)
        print(f'[EVAL] deboilerplated: TP={summary["TP"]} FP={summary["FP"]} FN={summary["FN"]} TN={summary["TN"]}',flush=True)
    print(f'[DONE] Summary: {OUT / "stage4_variant_summary.csv"}',flush=True)
    print(f'[DONE] Report: {OUT / "stage4_comparison_report.md"}',flush=True)
    print('[CAUTION] Metrics are conditional on the stratified sample, not population-wide recall or precision.',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'[ERROR] {exc}',file=sys.stderr,flush=True)
        raise
