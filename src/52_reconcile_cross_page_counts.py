#!/usr/bin/env python3
"""Task 2: Reconcile baseline manuscript cross-page and country counts.

Read-only. Does NOT rebuild families, rescore candidates, or alter the ACL paper.
Requires original src/26_lineage_foundation.py, src/42_reestimate_downstream.py,
and saved baseline/cleaned members in the existing project.
Run: python src/52_reconcile_cross_page_counts.py
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'outputs' / 'anxiety_lineages'
OUT = BASE / 'reviewer_experiments' / 'cross_page_reconciliation'
POSTS = ROOT / 'data' / 'processed' / 'posts_clean.parquet'
MAPPING = BASE / 'minhash' / 'text_id_mapping.parquet'
MEMBERS = {
    'baseline': BASE / 'combined_cohesion_repair' / 'repaired_lineage_members.parquet',
    'deboilerplated': BASE / 'stage4_full_ablation' / 'deboilerplated' / 'combined_cohesion_repair' / 'repaired_lineage_members.parquet',
}
PAPER = {'baseline': {'families': 21441, 'cross_page': 8460, 'cross_country': 3041}}
SCRIPT42 = ROOT / 'src' / '42_reestimate_downstream.py'


def module42():
    if not SCRIPT42.exists():
        raise FileNotFoundError(f'Missing {SCRIPT42}. Do not repeat normalization guesses.')
    spec = importlib.util.spec_from_file_location('verified_script42_for_audit', SCRIPT42)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, '_producer_normalized_text'):
        raise RuntimeError('Your Script 42 lacks _producer_normalized_text. Use the working version that produced the verified 282,192/282,192 join.')
    return module


def get_posts():
    import pyarrow.parquet as pq
    if not POSTS.exists() or not MAPPING.exists():
        raise FileNotFoundError('Existing posts_clean.parquet or text_id_mapping.parquet missing; no files modified.')
    available = pq.ParquetFile(POSTS).schema_arrow.names
    columns = [c for c in [
        'Message', 'Page Name', 'User Name', 'Facebook Id', 'Page ID',
        'Page Id', 'Page URL', 'URL', 'Page Admin Top Country', 'Page Category'
    ] if c in available]
    if 'Message' not in columns or 'Page Name' not in columns:
        raise RuntimeError(f'Required Message/Page Name missing. Found: {available}')
    posts = pd.read_parquet(POSTS, columns=columns)
    mapping = pd.read_parquet(MAPPING, columns=['text_id','normalized_text','post_count'])
    mapping = mapping.sort_values('text_id').reset_index(drop=True)
    if not np.array_equal(mapping.text_id.to_numpy(dtype=np.int64), np.arange(len(mapping))):
        raise RuntimeError('Noncontiguous mapping text IDs; refusing to join.')
    normalizer = module42()
    norm = normalizer._producer_normalized_text(posts)
    if norm is None:
        raise RuntimeError('Could not apply exact original Script 26 normalizer; no estimates calculated.')
    idx = pd.Index(mapping.normalized_text.astype(str))
    ids = idx.get_indexer(norm.fillna('').astype(str))
    matched = int((ids >= 0).sum())
    counts = np.bincount(ids[ids >= 0], minlength=len(mapping))
    wrong = int(np.count_nonzero(counts != mapping.post_count.to_numpy(dtype=np.int64)))
    expected = int(mapping.post_count.sum())
    print(f'[JOIN] exact-normalizer matched={matched:,}/{expected:,}; count mismatches={wrong:,}', flush=True)
    if matched != expected or wrong:
        raise RuntimeError('Post-to-text mapping is not exact. Stopping before producing comparison.')
    posts = posts.iloc[np.flatnonzero(ids >= 0)].copy()
    posts['text_id'] = ids[ids >= 0].astype(np.int64)
    return posts, mapping, columns


def family_members(variant, mapping):
    path = MEMBERS[variant]
    if not path.exists():
        raise FileNotFoundError(f'Saved {variant} family memberships not found: {path}')
    members = pd.read_parquet(path, columns=['text_id','repaired_lineage_id'])
    if members.text_id.duplicated().any():
        raise RuntimeError(f'Duplicate membership text ID: {path}')
    members = members.rename(columns={'repaired_lineage_id': 'lineage'}).copy()
    members['lineage'] = variant + '_M_' + members.lineage.astype(str)
    exact = mapping.loc[(mapping.post_count >= 2) & (~mapping.text_id.isin(members.text_id)), ['text_id']].copy()
    exact['lineage'] = variant + '_E_' + exact.text_id.astype(str)
    return pd.concat([members[['text_id','lineage']], exact], ignore_index=True)


def canonical(values):
    s = values.astype('string').str.strip()
    # These are missing-value sentinels, not real pages/countries.
    return s.mask(s.isna() | s.str.lower().isin(['', 'unknown', 'null', 'none', 'nan', 'n/a', 'na', '-', 'not specified']))


def measure(p, variant, column, mode, paper_count=None):
    if column not in p:
        return None, None
    key = canonical(p[column])
    if mode == 'missing_as_unknown':
        key = key.fillna('__UNKNOWN__')
    elif mode == 'ignore_missing':
        pass
    elif mode == 'raw_values':
        key = p[column].astype('string').fillna('__UNKNOWN__')
    elif mode == 'casefold':
        key = key.str.casefold()
    else:
        raise ValueError(mode)
    distinct = pd.DataFrame({'lineage': p['lineage'].to_numpy(), 'key': key.to_numpy()}).groupby('lineage', sort=False)['key'].nunique(dropna=True)
    cross = distinct.ge(2)
    total = int(cross.sum())
    out = {'variant': variant, 'field': column, 'rule': mode, 'families': len(distinct),
           'cross_families': total, 'cross_share': total / len(distinct),
           'missing_posts': int(key.isna().sum()) if mode != 'missing_as_unknown' else int((key == '__UNKNOWN__').sum())}
    if paper_count is not None:
        out['delta_vs_manuscript'] = total - paper_count
    return out, cross


def inspect_original_propagation_script():
    paths = [ROOT / 'src' / '36_lineage_propagation.py', ROOT / 'src' / '45_generate_acl_figures.py']
    lines = []
    for p in paths:
        if not p.exists():
            continue
        src = p.read_text(encoding='utf-8-sig', errors='replace').splitlines()
        for i, row in enumerate(src, 1):
            lower = row.lower()
            if any(v in lower for v in ('cross_page','cross-page','cross_country','page name','user name','page_admin', 'nunique', 'unique_pages', 'page_id')):
                lines.append(f'{p.relative_to(ROOT)}:{i}: {row.strip()[:240]}')
    if lines:
        (OUT / 'original_method_code_lines.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('[SOURCE] Saved relevant lines from original propagation/figure scripts',flush=True)
    else:
        print('[SOURCE] Original propagation/figure code not found or no matching lines. Audit definitions remain exploratory.', flush=True)


def main():
    print('[TASK 2] Cross-page and country discrepancy audit; read-only; no reconstruction',flush=True)
    posts, mapping, available = get_posts()
    print(f'[POST FIELDS] {available}',flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    comparisons = []
    for variant in ('baseline', 'deboilerplated'):
        members = family_members(variant, mapping)
        p = posts.merge(members, on='text_id', how='inner', validate='many_to_one')
        total = int(members.lineage.nunique())
        print(f'[FAMILIES] {variant}: families={total:,}, linked_posts={len(p):,}',flush=True)
        if variant == 'baseline' and total != PAPER['baseline']['families']:
            raise RuntimeError(f'Baseline family count {total:,} != manuscript 21,441. Wrong snapshot or membership!')
        page_options = ['Page Name','User Name','Page ID','Page Id','Page URL']
        # Facebook Id may be a post identifier: report only, not a default page key.
        for col in page_options:
            if col not in p: continue
            for mode in ('raw_values', 'missing_as_unknown', 'ignore_missing', 'casefold'):
                row, cross = measure(p, variant, col, mode,
                    PAPER['baseline']['cross_page'] if variant == 'baseline' else None)
                report.append(row)
                comparisons.append((variant, col, mode, cross))
                print(f'[PAGE] {variant} {col}/{mode}: {row["cross_families"]:,} cross-page families',flush=True)
        if 'Page Name' in p and 'User Name' in p:
            # Alternative page ID combines display name and username; a change is
            # only a diagnostic, not evidence that this is the right identifier.
            page_name = canonical(p['Page Name'])
            username = canonical(p['User Name'])
            p['_name_plus_user'] = page_name.fillna('?') + ' :: ' + username.fillna('?')
            row, cross = measure(p,variant,'_name_plus_user','ignore_missing',
                PAPER['baseline']['cross_page'] if variant=='baseline' else None)
            row['field']='Page Name + User Name'
            report.append(row)
            comparisons.append((variant, 'Page Name + User Name', 'ignore_missing', cross))
            print(f'[PAGE] {variant} combined name+username: {row["cross_families"]:,}',flush=True)
        if 'Facebook Id' in p:
            unique_rate = p['Facebook Id'].nunique(dropna=True)/max(1,len(p))
            print(f'[FIELD] Facebook Id unique per linked post={unique_rate:.2%}; cannot assume page ID',flush=True)
        if 'Page Admin Top Country' in p:
            for mode in ('raw_values','missing_as_unknown','ignore_missing','casefold'):
                row,cross = measure(p,variant,'Page Admin Top Country',mode,
                      PAPER['baseline']['cross_country'] if variant=='baseline' else None)
                report.append(row)
                comparisons.append((variant,'Page Admin Top Country',mode,cross))
                print(f'[COUNTRY] {variant} {mode}: {row["cross_families"]:,} cross-country families',flush=True)
        # Compare family-level changes caused by handling missing metadata differently.
        d={(col,mode):cross for v,col,mode,cross in comparisons if v==variant}
        primary=d.get(('Page Name','raw_values'))
        country_ignore=d.get(('Page Admin Top Country','ignore_missing'))
        if primary is not None:
            for col, mode in [('User Name','ignore_missing'),('Page Name + User Name','ignore_missing')]:
                alt=d.get((col,mode))
                if alt is None:continue
                change = primary.index.union(alt.index)
                primary_re=primary.reindex(change,fill_value=False)
                alt_re=alt.reindex(change,fill_value=False)
                subset = change[primary_re != alt_re]
                if len(subset):
                    pd.DataFrame({'lineage':subset,
                                  'PageName_cross':primary_re.loc[subset].to_numpy(),
                                  f'{col}_cross':alt_re.loc[subset].to_numpy()}).to_csv(
                                   OUT/f'{variant}_page_definition_disagreements_{col.replace(" ","_").replace("+","and")}.csv',index=False)
        if country_ignore is not None:
            unknown=d.get(('Page Admin Top Country','missing_as_unknown'))
            if unknown is not None:
                delta=unknown[unknown!=country_ignore]
                delta.rename('missing_country_changes_cross_flag').to_csv(OUT/f'{variant}_missing_country_affected_families.csv')
    summary=pd.DataFrame(report)
    summary.to_csv(OUT/'cross_page_definition_comparison.csv',index=False)
    matches=summary[(summary.variant=='baseline') & (summary.delta_vs_manuscript==0)]
    print('[TARGET] Original manuscript: 8,460 cross-page; 3,041 cross-country',flush=True)
    if len(matches):
        print('[TARGET MATCHES]',matches[['field','rule','cross_families']].to_dict('records'),flush=True)
    else:
        print('[TARGET] No tested metadata definition exactly reproduces manuscript counts. Original propagation source must be compared before choosing a correction.',flush=True)
    inspect_original_propagation_script()
    print(f'[DONE] {OUT / "cross_page_definition_comparison.csv"}',flush=True)
    print('[CAUTION] An exact numerical match does not by itself establish the historical identifier definition.',flush=True)

if __name__=='__main__':main()
