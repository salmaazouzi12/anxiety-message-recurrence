#!/usr/bin/env python3
"""READ-ONLY audit of family/text disjointness for the finished human benchmark.

Put this file directly in anxiety-research/src/. Run from project root:
    python src/51_verify_holdout_disjointness.py

No graph reconstruction, no annotation changes, no manuscript edits.
Checks the 400 development pair IDs, 120 new pair IDs, 40 new family triplets
under BOTH the baseline and cleaned reconstruction memberships. Optionally
checks the original 100 development family samples if their annotated text file
is present (family_annotations_completed.csv); use --old-families PATH otherwise.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
BASE = ROOT / 'outputs' / 'anxiety_lineages'
BLIND = BASE / 'reviewer_experiments' / 'blind_validation'
OUT = BASE / 'reviewer_experiments' / 'holdout_disjointness_audit'


def load_key(path, required):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required key: {path}. Do not regenerate Script 45 keys.')
    d = pd.read_csv(path, encoding='utf-8-sig')
    if not set(required).issubset(d.columns):
        raise ValueError(f'{path}: expected columns {required}; found {d.columns.tolist()}')
    return d


def membership(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved reconstruction: {path}')
    d = pd.read_parquet(path, columns=['text_id', 'repaired_lineage_id'])
    if d.text_id.isna().any() or d.text_id.duplicated().any():
        raise ValueError(f'Invalid or duplicate text IDs in {path}')
    return dict(zip(d.text_id.astype(int), d.repaired_lineage_id.astype(str)))


def ids_from_family_key(value):
    if pd.isna(value):
        raise ValueError('Blank family triplet in private key')
    v = str(value)
    ids = [int(t.strip()) for t in v.split(',') if t.strip()]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError(f'Expected exactly 3 distinct text IDs; found {value!r}')
    return ids


def family_sets(ids, members):
    return {members[t] for t in ids if t in members}


def locate_csv(label, preferred, filename_patterns, required, expected_rows):
    """Find pre-existing audit CSVs within standard project locations.

    Prefer the canonical path, but inspect src/ subfolders and output folders;
    tolerate Windows-style downloaded names such as pair_sampling_key(1).csv.
    Check structure AND number of rows, never silently substitute another key.
    """
    options = []
    seen = set()
    for folder in (SRC, ROOT, BASE / 'reviewer_experiments', BASE):
        if not folder.is_dir():
            continue
        for pattern in filename_patterns:
            matches = folder.glob(pattern) if folder == ROOT else folder.rglob(pattern)
            for path in matches:
                if not path.is_file():
                    continue
                norm = str(path.resolve()).casefold()
                if norm in seen:
                    continue
                seen.add(norm)
                options.append(path)
    preferred_set = {str(p.resolve()).casefold() for p in preferred}
    options.sort(key=lambda p:(str(p.resolve()).casefold() not in preferred_set,
                               len(p.parts), str(p).casefold()))
    valid=[]
    for path in options:
        try:
            df=pd.read_csv(path, encoding='utf-8-sig')
            if len(df)!=expected_rows or not set(required).issubset(df.columns):
                continue
            valid.append((path,df))
        except (OSError,UnicodeError,pd.errors.ParserError):
            continue
    if not valid:
        nearby='; '.join(str(p.relative_to(ROOT)) for p in options[:12]) or 'none'
        raise FileNotFoundError(
            f'{label}: no matching {expected_rows}-row key with columns {required} found. '
            f'Files checked: {nearby}. Existing keys were not altered.')
    path, df = valid[0]
    # Warn if the same logical key name occurs with differing contents.
    for other,other_df in valid[1:]:
        col=required
        if not df[col].reset_index(drop=True).equals(other_df[col].reset_index(drop=True)):
            print(f'[WARNING] Different valid {label} also exists at {other}; '
                  'using first selected key. Confirm which was used to sample annotations.', flush=True)
    print(f'[FOUND] {label}: {path} ({len(df)} rows)', flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-families', type=Path, default=None,
                    help='Original 100-family annotation CSV, if stored outside project src')
    args = ap.parse_args()
    if SRC.name.lower() != 'src' or not BASE.is_dir():
        raise RuntimeError(f'Place this file in your MAIN project src/ folder: {SRC}')
    OUT.mkdir(parents=True, exist_ok=True)

    dev = locate_csv('Original 400-pair development key',
                     [SRC/'pair_sampling_key.csv', ROOT/'pair_sampling_key.csv'],
                     ['pair_sampling_key*.csv'],
                     ['text_id_a','text_id_b'], 400)
    pairs = locate_csv('Private new 120-pair key',
                      [BLIND/'PRIVATE_pair_sampling_key_DO_NOT_SHARE.csv'],
                      ['PRIVATE_pair_sampling_key_DO_NOT_SHARE*.csv'],
                      ['item_id','text_id_a','text_id_b'], 120)
    fams = locate_csv('Private new 40-family key',
                     [BLIND/'PRIVATE_family_sampling_key_DO_NOT_SHARE.csv'],
                     ['PRIVATE_family_sampling_key_DO_NOT_SHARE*.csv'],
                     ['item_id','text_ids'], 40)
    if len(dev) != 400 or len(pairs) != 120 or len(fams) != 40:
        raise ValueError(f'Unexpected sample sizes: dev={len(dev)} (expected 400), '
                         f'heldout_pairs={len(pairs)} (expected 120), '
                         f'heldout_families={len(fams)} (expected 40). Stop and check original keys.')
    if pairs.item_id.duplicated().any() or fams.item_id.duplicated().any():
        raise ValueError('Duplicate new sample IDs')

    old_ids = set(dev.text_id_a.astype(int)) | set(dev.text_id_b.astype(int))
    items = {}
    for r in pairs.itertuples(index=False):
        ids = [int(r.text_id_a), int(r.text_id_b)]
        if ids[0] == ids[1]:
            raise ValueError(f'Pair {r.item_id}: identical source text IDs')
        items[f'pair:{r.item_id}'] = ids
    for r in fams.itertuples(index=False):
        items[f'family:{r.item_id}'] = ids_from_family_key(r.text_ids)

    paths = {
        'baseline': BASE/'combined_cohesion_repair'/'repaired_lineage_members.parquet',
        'cleaned': BASE/'stage4_full_ablation'/'deboilerplated'/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
    }
    mems = {name: membership(path) for name,path in paths.items()}
    print(f'[INPUT] development={len(dev)} pairs, new={len(pairs)} pairs + {len(fams)} triplets', flush=True)
    print('[INPUT] Reading saved baseline + cleaned memberships; no reconstruction', flush=True)

    # Aggregate IDs of all development sample posts and every new sample item.
    new_ids = {t for ids in items.values() for t in ids}
    examples = []
    checks = {}
    shared = sorted(old_ids & new_ids)
    checks['direct_text_id_overlap_with_dev400'] = len(shared)
    for t in shared:
        examples.append({'check':'direct_text_id_overlap_with_dev400','text_id':t,'item_a':'development_400','item_b':'heldout'})

    owners_by_text = defaultdict(set)
    for item,ids in items.items():
        for t in ids:
            owners_by_text[t].add(item)
    repeat_text = {t:sorted(v) for t,v in owners_by_text.items() if len(v)>1}
    checks['reused_text_ids_across_new_items'] = len(repeat_text)
    for t,owners in repeat_text.items():
        examples.append({'check':'reused_text_ids_across_new_items','text_id':t,'item_a':owners[0],'item_b':owners[1]})

    for name,mem in mems.items():
        dev_fams = family_sets(old_ids,mem)
        new_fams = family_sets(new_ids,mem)
        common = sorted(dev_fams & new_fams)
        checks[f'{name}_families_shared_dev400_vs_holdout'] = len(common)
        for fid in common:
            examples.append({'check':f'{name}_families_shared_dev400_vs_holdout','family_id':fid,'item_a':'development_400','item_b':'heldout'})
        owners_by_family = defaultdict(set)
        for item,ids in items.items():
            for fid in family_sets(ids,mem):
                owners_by_family[fid].add(item)
        collisions = {fid:sorted(owners) for fid,owners in owners_by_family.items() if len(owners)>1}
        checks[f'{name}_families_reused_across_new_items'] = len(collisions)
        for fid,owners in collisions.items():
            examples.append({'check':f'{name}_families_reused_across_new_items','family_id':fid,'item_a':owners[0],'item_b':owners[1]})
        print(f'[CHECK] {name}: development-family overlaps={len(common)}, '
              f'family reuse between new items={len(collisions)}',flush=True)

    # The older study also annotated 100 family triplets. Their *text* can be
    # checked if that older annotation file is available; numeric lineage labels
    # such as M_425 are NOT the same ID system as repaired_lineage_id.
    old_family_path = args.old_families
    if old_family_path is None:
        choices = [SRC/'family_annotations_completed.csv', ROOT/'family_annotations_completed.csv']
        choices += sorted(SRC.rglob('family_annotations_completed*.csv'))
        choices += sorted((BASE/'reviewer_experiments').rglob('family_annotations_completed*.csv'))
        old_family_path = next((p for p in choices if p.is_file()),None)
    old100 = {'status':'NOT_CHECKED: old 100-family text file absent',
              'source':None, 'rows':None, 'matched_text_samples':None,
              'unresolved_text_samples':None}
    if old_family_path is not None:
        if not old_family_path.is_absolute():
            old_family_path = ROOT / old_family_path
        old = pd.read_csv(old_family_path,encoding='utf-8-sig')
        cols = [c for c in ('first_text','last_text','farthest_text') if c in old.columns]
        if not cols:
            raise ValueError(f'Old family file {old_family_path} lacks first_text/last_text/farthest_text')
        mp = pd.read_parquet(BASE/'minhash'/'text_id_mapping.parquet',columns=['text_id','normalized_text'])
        # Exact mapping only: fuzzy guesses could falsely certify family disjointness.
        text_to_ids = defaultdict(list)
        for t,s in zip(mp.text_id,mp.normalized_text):
            if pd.notna(s):text_to_ids[str(s)].append(int(t))
        old_representative_ids = set()
        unresolved = []
        for ix,r in old.reset_index(drop=True).iterrows():
            for c in cols:
                v = r[c]
                if pd.isna(v):continue
                m = text_to_ids.get(str(v),[])
                if len(m)==1:old_representative_ids.add(m[0])
                else:unresolved.append({'row':ix,'field':c,'matches':len(m)})
        old100={'status':'PARTIAL' if unresolved else 'RESOLVED_REPRESENTATIVES_ONLY',
                'source':str(old_family_path),'rows':len(old),
                'matched_text_samples':len(old_representative_ids),
                'unresolved_text_samples':len(unresolved),
                'warning':'Even exact match of the three representative texts does not prove disjointness from every member of those 100 older families.'}
        checks['old100_representative_text_overlap'] = len(old_representative_ids & new_ids)
        for name,mem in mems.items():
            n=len(family_sets(old_representative_ids,mem) & family_sets(new_ids,mem))
            checks[f'{name}_old100_representative_family_overlap']=n
        if unresolved:
            pd.DataFrame(unresolved).to_csv(OUT/'unresolved_old100_representatives.csv',index=False)
        print(f'[CHECK] older 100-family representatives: matched={len(old_representative_ids)}, '
              f'unresolved={len(unresolved)}, direct overlap={checks["old100_representative_text_overlap"]}',flush=True)
    else:
        print('[NOTE] Original 100-family annotations unavailable; cannot certify full disjointness from them.',flush=True)

    # Report includes no blind text or annotator labels; do not disclose private keys.
    pd.DataFrame(examples,columns=['check','text_id','family_id','item_a','item_b']).to_csv(
        OUT/'overlap_examples_PRIVATE.csv',index=False)
    core_keys=[k for k in checks if 'old100' not in k]
    core_ok=all(checks[k]==0 for k in core_keys)
    old100_checks=[k for k in checks if 'old100' in k]
    old100_ok=bool(old100_checks) and all(checks[k]==0 for k in old100_checks)
    report={
        'audit':'holdout_family_disjointness',
        'development_400_pairs':len(dev),'new_heldout_pairs':len(pairs),
        'new_heldout_family_triplets':len(fams),'development_unique_text_ids':len(old_ids),
        'new_unique_text_ids':len(new_ids),'checks':checks,'old_100_families':old100,
        'result_dev400_vs_new':'PASS' if core_ok else 'FAIL',
        'result_full_prior_annotation_disjointness':(
            'NOT FULLY VERIFIED (old 100 families unavailable/unresolved)' if not old100_ok
            else 'NO DETECTED OVERLAP IN OLD FAMILY REPRESENTATIVES; other members not verified'),
        'note':'Disjointness concerns graph components, not a random probability sample. '
               'A new human audit cannot by itself establish population accuracy.'
    }
    (OUT/'holdout_disjointness_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('[RESULT] 400-pair development vs new samples:',report['result_dev400_vs_new'],flush=True)
    print('[RESULT] Older family-sample check:',report['result_full_prior_annotation_disjointness'],flush=True)
    print('[SAVED]',OUT/'holdout_disjointness_report.json',flush=True)
    if not core_ok:
        print('[WARNING] Overlap found. Do NOT describe these samples as family-disjoint.',flush=True)
        sys.exit(2)

if __name__=='__main__':main()
