#!/usr/bin/env python3
"""Score the FINAL independent human annotations against saved lineage decisions.

Usage (from anxiety-research root):
    python src/50_evaluate_final_human_validation.py

Requires original PRIVATE sampling keys saved by Script 45 and saved baseline /
de-boilerplated membership Parquet files. Does NOT rebuild any lineages, fit
thresholds, change human labels, or create consensus labels for disagreements.

Outputs: outputs/anxiety_lineages/reviewer_experiments/human_validation_results/
"""
from __future__ import annotations
import json
from pathlib import Path
from math import sqrt
from collections import Counter
import pandas as pd
import numpy as np

# This script is deliberately for the MAIN project src folder only.
# Save it as anxiety-research/src/50_evaluate_final_human_validation.py.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if SCRIPT_DIR.name.lower() != 'src' or not (ROOT/'outputs'/'anxiety_lineages').is_dir():
    raise RuntimeError(
        'Wrong script location. Place 50_evaluate_final_human_validation.py '+
        'DIRECTLY in anxiety-research\\src, alongside the four annotation CSVs. '+
        'Run: python src/50_evaluate_final_human_validation.py'
    )
BASE = ROOT/'outputs'/'anxiety_lineages'
BLIND = BASE/'reviewer_experiments'/'blind_validation'
OUT = BASE/'reviewer_experiments'/'human_validation_results'
BASE_MEM = BASE/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
CLEAN_MEM = BASE/'stage4_full_ablation'/'deboilerplated'/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
# Read checked A/B annotation CSVs ONLY from the MAIN src folder.
INPUT_DIRS = [SCRIPT_DIR]
print(f'[PROJECT] {ROOT}',flush=True)
print(f'[INPUTS] Checked annotation CSVs in {SCRIPT_DIR}',flush=True)

SOURCES={
 'pairs_A':['annotator_A_pairs_UTF8_restored.csv','blinded_pairs_annotator_A.csv'],
 'pairs_B':['annotated_pairs_B.csv','blinded_pairs_annotator_B.csv'],
 'families_A':['annotator_A_families_UTF8_restored.csv','blinded_families_annotator_A.csv'],
 'families_B':['annotated_families_B.csv','blinded_families_annotator_B.csv'],
}
PAIR_LABELS={'SAME','RELATED','UNRELATED'}
FAM_LABELS={'COHERENT','PARTIALLY COHERENT','NOT COHERENT','NOT SURE'}
# Only correct unambiguous label-spelling errors; do not alter any semantic judgment.
TYPOS={'REALTED':'RELATED','RELTED':'RELATED','UNRRELATED':'UNRELATED',
       'PARTIALLY_COHERENT':'PARTIALLY COHERENT','NOT_COHERENT':'NOT COHERENT',
       'NOT_SURE':'NOT SURE'}

def candidate_paths(kind):
    """Look for the correct file by CONTENT, not merely by its filename.

    Some CSV exports were mistakenly copied under the opposite annotation file
    name. The source folder is a convenience; the PRIVATE key item IDs and the
    legal label vocabulary are the authority for selecting annotations.
    """
    locations=INPUT_DIRS
    preferred=SOURCES[kind]
    # NEVER substitute B's file for A (or vice versa): that would fabricate agreement.
    # Wrongly named or incomplete files are skipped, not re-assigned.
    all_names=preferred
    seen=set()
    # Prefer the supplied, validated package folder before stray src copies.
    for loc in locations:
        for name in all_names:
            p=loc/name
            if p not in seen and p.is_file():
                seen.add(p)
                yield p

def load_labels(kind,valid,expected_ids):
    expected_ids=set(map(str,expected_ids))
    checked=[]
    for p in candidate_paths(kind):
        try:
            d=pd.read_csv(p,dtype={'item_id':'string'},encoding='utf-8-sig')
            if not {'item_id','label'}.issubset(d.columns):
                checked.append(f'{p}: missing item_id or label columns')
                continue
            if d.item_id.isna().any() or d.item_id.duplicated().any():
                checked.append(f'{p}: missing or duplicate item IDs')
                continue
            actual=set(d.item_id.astype(str))
            if actual!=expected_ids:
                # This catches the user's F0030-in-a-pairs-file failure.
                example=next(iter(sorted(actual-expected_ids)), 'none')
                checked.append(f'{p}: wrong set of item IDs ({len(actual)} found, '
                               f'{len(expected_ids)} expected, unexpected={example})')
                continue
            orig=d.label.fillna('').astype(str).str.strip().str.upper()
            fixed=orig.replace(TYPOS)
            wrong=d.loc[~fixed.isin(valid),'item_id'].tolist()
            if wrong:
                checked.append(f'{p}: {len(wrong)} missing/invalid labels, first IDs={wrong[:5]}')
                continue
            changes=int((orig!=fixed).sum())
            print(f'[ANNOTATOR] {kind}: {len(d)} valid labels from '
                  f'{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}; '
                  f'spelling fixes={changes}',flush=True)
            return pd.DataFrame({'item_id':d.item_id.astype(str),
                                 'label_'+kind[-1]:fixed})
        except (UnicodeError,pd.errors.ParserError) as exc:
            checked.append(f'{p}: cannot read CSV ({exc})')
    detail='\n - '.join(checked) if checked else 'No candidate files found'
    raise ValueError(f'No valid {kind} annotation file found with exactly '
                     f'{len(expected_ids)} required item IDs and completed labels.\n'
                     f'Checked:\n - {detail}\n'
                     'Put the FOUR completed annotation CSV files directly in the MAIN project src folder.')

def read_key(filename):
    p=BLIND/filename
    if not p.exists():
        raise FileNotFoundError(f'PRIVATE sampling key missing: {p}. The key was created by Script 45. Do NOT rerun Script 45: that would generate different item IDs. Restore the original private key.')
    d=pd.read_csv(p,dtype={'item_id':'string'})
    if 'item_id' not in d or d.item_id.duplicated().any():raise ValueError(f'Invalid private key: {p}')
    return d

def join_annotations(kind,valid,key):
    a=load_labels(kind+'_A',valid,key.item_id); b=load_labels(kind+'_B',valid,key.item_id)
    if set(a.item_id)!=set(b.item_id) or set(a.item_id)!=set(key.item_id):
        raise ValueError(f'{kind}: A/B IDs or private sampling key do not match exactly; refused unsafe evaluation')
    d=key.merge(a,on='item_id',validate='one_to_one').merge(b,on='item_id',validate='one_to_one')
    d['agreement']=d.label_A==d.label_B
    d['consensus_label']=d.label_A.where(d.agreement,pd.NA)
    return d

def read_members(path):
    if not path.exists():raise FileNotFoundError(f'Repaired membership not found: {path}')
    d=pd.read_parquet(path,columns=['text_id','repaired_lineage_id'])
    if d.text_id.duplicated().any():raise ValueError(f'Duplicate text_id in membership {path}')
    return dict(zip(d.text_id.astype(int),d.repaired_lineage_id.astype(str)))

def pair_pred(d,mem):
    a=d.text_id_a.astype(int).map(mem)
    b=d.text_id_b.astype(int).map(mem)
    return (a.notna() & b.notna() & (a==b)).astype(int)

def family_pred(ids,mem):
    labels=[mem.get(int(t)) for t in ids]
    # Unlinked texts represent separate singleton components, not one shared NA label.
    labels=[v if v is not None else f'UNLINKED_{i}_{t}' for i,(v,t) in enumerate(zip(labels,ids))]
    unique=len(set(labels))
    return 'COHERENT' if unique==1 else 'PARTIALLY COHERENT' if unique==2 else 'NOT COHERENT'

def wilson(success,n,z=1.959963984540054):
    if not n:return [None,None]
    p=success/n; den=1+z*z/n
    ctr=(p+z*z/(2*n))/den
    half=z/den*sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [ctr-half,ctr+half]

def binary_metrics(gold,pred):
    gold=np.asarray(gold,dtype=bool);pred=np.asarray(pred,dtype=bool)
    tp=int((gold&pred).sum());fp=int((~gold&pred).sum());fn=int((gold&~pred).sum());tn=int((~gold&~pred).sum())
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None
    return {'n':len(gold),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'accuracy':(tp+tn)/len(gold) if len(gold) else None,
            'precision':precision,'recall':recall,'f1':f1,
            'accuracy_wilson95':wilson(tp+tn,len(gold))}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print(f'[MEMBERSHIPS] Baseline: {BASE_MEM}', flush=True)
    print(f'[MEMBERSHIPS] Cleaned:  {CLEAN_MEM}', flush=True)
    base=read_members(BASE_MEM); clean=read_members(CLEAN_MEM)
    kp=read_key('PRIVATE_pair_sampling_key_DO_NOT_SHARE.csv')
    if not {'text_id_a','text_id_b'}.issubset(kp.columns):raise ValueError('Private pair key missing text IDs')
    pairs=join_annotations('pairs',PAIR_LABELS,kp)
    pairs['baseline_same_reconstructed']=pair_pred(pairs,base)
    pairs['cleaned_same_reconstructed']=pair_pred(pairs,clean)
    pairs['model_predictions_differ']=pairs.baseline_same_reconstructed!=pairs.cleaned_same_reconstructed
    results={}
    rows=[]
    for group_name,sub in [('annotator_A',pairs),('annotator_B',pairs),('annotators_agree',pairs[pairs.agreement])]:
        glabel=sub['consensus_label'] if group_name=='annotators_agree' else sub['label_A' if group_name=='annotator_A' else 'label_B']
        y=glabel.eq('SAME').to_numpy()
        for variant,col in [('baseline','baseline_same_reconstructed'),('deboilerplated','cleaned_same_reconstructed')]:
            met=binary_metrics(y,sub[col].to_numpy())
            rows.append({'reference':group_name,'variant':variant,**{k:v for k,v in met.items() if k!='accuracy_wilson95'},
                         'accuracy_CI95_low':met['accuracy_wilson95'][0],'accuracy_CI95_high':met['accuracy_wilson95'][1]})
            print(f"[PAIR EVAL] {group_name:18s} {variant:14s} n={met['n']:3d} TP={met['TP']:3d} FP={met['FP']:3d} FN={met['FN']:3d} TN={met['TN']:3d} F1={met['f1']}",flush=True)
        # Overall paired change in errors, NOT a cluster-independent significance claim.
        b=(sub.baseline_same_reconstructed.to_numpy()==y)
        c=(sub.cleaned_same_reconstructed.to_numpy()==y)
        results[group_name]={'n':len(sub),'baseline_correct_cleaned_wrong':int((b&~c).sum()),
                              'baseline_wrong_cleaned_correct':int((~b&c).sum()),
                              'model_pred_changed':int(sub.model_predictions_differ.sum())}
    # Human agreement describes human-human reliability, not model correctness.
    from sklearn.metrics import cohen_kappa_score
    pair_agree={'n':len(pairs),'agree':int(pairs.agreement.sum()),'agreement':float(pairs.agreement.mean()),
                'cohen_kappa':float(cohen_kappa_score(pairs.label_A,pairs.label_B,labels=['SAME','RELATED','UNRELATED']))}
    print('[PAIR HUMAN AGREEMENT]',json.dumps(pair_agree),flush=True)
    pairs[['item_id','sampling_stratum','label_A','label_B','agreement','consensus_label',
           'baseline_same_reconstructed','cleaned_same_reconstructed','model_predictions_differ']].to_csv(OUT/'pair_predictions_and_human_labels.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUT/'pair_model_metrics.csv',index=False)
    if 'sampling_stratum' in pairs:
        stratum_rows=[]
        for strata,subset in pairs.groupby('sampling_stratum',dropna=False):
            for tag,col in [('baseline','baseline_same_reconstructed'),('deboilerplated','cleaned_same_reconstructed')]:
                for referee,labelcol in [('A','label_A'),('B','label_B')]:
                    met=binary_metrics(subset[labelcol].eq('SAME').to_numpy(),subset[col].to_numpy())
                    stratum_rows.append({'sampling_stratum':strata,'annotator':referee,'variant':tag,**{k:v for k,v in met.items() if k!='accuracy_wilson95'}})
        pd.DataFrame(stratum_rows).to_csv(OUT/'pair_metrics_by_sampling_stratum.csv',index=False)

    kf=read_key('PRIVATE_family_sampling_key_DO_NOT_SHARE.csv')
    if 'text_ids' not in kf.columns:raise ValueError('Private family key missing text_ids')
    families=join_annotations('families',FAM_LABELS,kf)
    for variant,mem in [('baseline',base),('deboilerplated',clean)]:
        families[variant+'_family_prediction']=[family_pred([int(x.strip()) for x in ids.split(',')],mem) for ids in families.text_ids.astype(str)]
    frows=[]
    for ref,labelcol,subset in [('annotator_A','label_A',families),('annotator_B','label_B',families),
                                ('annotators_agree','consensus_label',families[families.agreement])]:
        eligible=subset[subset[labelcol].ne('NOT SURE') & subset[labelcol].notna()]
        for variant in ['baseline','deboilerplated']:
            predcol=variant+'_family_prediction'
            exact=int((eligible[labelcol]==eligible[predcol]).sum())
            full_true=eligible[labelcol].eq('COHERENT'); full_pred=eligible[predcol].eq('COHERENT')
            met=binary_metrics(full_true,full_pred)
            frows.append({'reference':ref,'variant':variant,'n_non_unsure':len(eligible),
                          'three_class_accuracy':exact/len(eligible) if len(eligible) else None,
                          'three_class_correct':exact,'predicted_coherent':int(full_pred.sum()),
                          'human_coherent':int(full_true.sum()),
                          'coherent_binary_f1':met['f1']})
            print(f'[FAMILY EVAL] {ref:18s} {variant:14s} non_unsure={len(eligible)} exact={exact}',flush=True)
    f_agree={'n':len(families),'agree':int(families.agreement.sum()),'agreement':float(families.agreement.mean()),
             'cohen_kappa':float(cohen_kappa_score(families.label_A,families.label_B,labels=sorted(FAM_LABELS)))}
    print('[FAMILY HUMAN AGREEMENT]',json.dumps(f_agree),flush=True)
    families[['item_id','label_A','label_B','agreement','consensus_label',
              'baseline_family_prediction','deboilerplated_family_prediction']].to_csv(OUT/'family_predictions_and_human_labels.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(frows).to_csv(OUT/'family_model_metrics.csv',index=False)
    summary={'pair_interannotator_agreement':pair_agree,'family_interannotator_agreement':f_agree,
             'paired_changes_by_reference':results,
             'interpretation_constraints':[
                 'The sample is deliberately stratified, and metrics are conditional on the sampled item mix.',
                 'These human annotations are a held-out family-disjoint round relative to the earlier 400-pair development sample, insofar as Script 45 sampling assumptions hold.',
                 'A/B labels were not overridden. Where they disagree, no consensus gold label is fabricated; each annotator and the agreed subset are evaluated separately.',
                 'All 40 triplets were sampled from cleaned reconstructed families, so they are not a balanced positive/negative family classification test.',
                 'Do not treat pair decisions sharing a family as statistically independent.',
                 'The private item-to-text key and IDs are never written to the public output files.'
             ]}
    (OUT/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'README_RESULTS.txt').write_text(
        'Reviewer validation outputs: pair_model_metrics.csv, pair_metrics_by_sampling_stratum.csv, '+
        'family_model_metrics.csv, validation_summary.json. No model retraining or reconstructed graph change. '+
        'A/B remain separate on disagreements. The family sample was drawn from cleaned graph families by construction.\n',encoding='utf-8')
    print(f'[DONE] Human-vs-model evaluation saved at {OUT}',flush=True)

if __name__=='__main__':main()
