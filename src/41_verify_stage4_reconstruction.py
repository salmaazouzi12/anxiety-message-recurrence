#!/usr/bin/env python3
"""Verify/rebuild Stage 4 de-boilerplated graph provenance without rescoring candidates.

From project root:
  python src/41_verify_stage4_reconstruction.py
  python src/41_verify_stage4_reconstruction.py --rebuild --quiet

Requires scripts 38 and 40, verified Stage 4 cleaned scores and the completed
cohesion variants. Rebuild archives ONLY Stage 4's de-boilerplated graph outputs.
It never modifies original project lineages or reruns candidate scoring.
"""
import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import json
import shutil
import sys


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def describe(p):
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds') if p.is_file() else 'MISSING'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rebuild', action='store_true', help='If stale, archive only experimental graph outputs and rerun Scripts 34/35')
    ap.add_argument('--force', action='store_true', help='With --rebuild, rebuild even if timestamps look current')
    ap.add_argument('--quiet', action='store_true', help='Redirect original script console logs to Stage 4 run_logs.txt')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    stage4=import_file(root/'src'/'38_stage4_faithful_ablation.py','stage4_from_38')
    fast=import_file(root/'src'/'40_stage4_boilerplate_fast.py','stage4_from_40')
    variant=stage4.OUT/'deboilerplated'
    mapping=variant/'text_id_mapping_clean.parquet'
    scored=list((variant/'lexical_scoring').glob('scored_pairs_fast_*.parquet'))
    scored += list((variant/'combined_retrieval'/'novel_e5_scored').glob('scored_semantic_fast_*.parquet'))
    if not mapping.is_file() or not scored:
        raise FileNotFoundError('Missing cleaned mapping or verified fast scoring files. Finish Script 40 first.')
    fast.ensure_no_legacy_partial(stage4)
    missing_markers=[p for p in scored if not p.with_name(p.name+'.done.json').is_file()]
    if missing_markers:
        raise RuntimeError(f'Missing scoring completion markers: {missing_markers[:3]}')
    inputs=scored+[mapping]
    newest_input=max(inputs,key=lambda p:p.stat().st_mtime_ns)
    combined=variant/'combined_lineages'/'combined_lineage_members.parquet'
    members=variant/'combined_cohesion_repair'/'repaired_lineage_members.parquet'
    inventory=variant/'combined_cohesion_repair'/'repaired_lineage_inventory.parquet'
    graph_current=(combined.is_file() and members.is_file() and inventory.is_file()
                   and combined.stat().st_mtime_ns>=newest_input.stat().st_mtime_ns
                   and members.stat().st_mtime_ns>=combined.stat().st_mtime_ns
                   and inventory.stat().st_mtime_ns>=combined.stat().st_mtime_ns)
    print(f'[CHECK] {len(scored):,} completed cleaned-score shards')
    print(f'[CHECK] Newest cleaned input: {newest_input.name} ({describe(newest_input)})')
    print(f'[CHECK] Graph construction: {describe(combined)}')
    print(f'[CHECK] Graph repair: {describe(members)}')
    print(f'[STATUS] {"CURRENT by file timestamps" if graph_current else "STALE or missing graph output"}')
    if not args.rebuild:
        if not graph_current:
            print('[ACTION] Rebuild graph without rescoring: python src/41_verify_stage4_reconstruction.py --rebuild --quiet')
        else:
            print('[NOTE] Timestamps are a preliminary provenance check, not proof of identical methodology or outcomes.')
        return
    if graph_current and not args.force:
        print('[SKIP] No rebuild needed. Add --force to rebuild deliberately.')
        return
    # Move only this variant's experimental graph directories, never the saved baseline.
    archive_root=variant/'previous_graph_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for folder_name in ('combined_lineages','combined_cohesion_repair'):
        folder=variant/folder_name
        if folder.exists():
            archive_root.mkdir(parents=True,exist_ok=True)
            shutil.move(str(folder),str(archive_root/folder_name))
            print(f'[ARCHIVED] {folder_name} to {archive_root}')
    stage4.GENERATED.mkdir(parents=True,exist_ok=True)
    print('[BUILD] Original Script 34 on verified cleaned scores (no rescoring)...',flush=True)
    stage4.run_original_variant(stage4.generate_build_clean('deboilerplated'),verbose=not args.quiet)
    if not combined.is_file():
        raise RuntimeError('Script 34 did not produce combined_lineage_members.parquet')
    print('[REPAIR] Original Script 35 on rebuilt graph...',flush=True)
    stage4.run_original_variant(stage4.generate_repair('deboilerplated',source='deboilerplated'),verbose=not args.quiet)
    if not members.is_file() or not inventory.is_file():
        raise RuntimeError('Script 35 did not produce both required outputs')
    fast.finalize(stage4,members)
    provenance={
        'experiment':'deboilerplated',
        'candidate_pool':'existing original MinHash and E5 candidates',
        'cleaned_score_file_count':len(scored),
        'newest_cleaned_input':str(newest_input),
        'newest_cleaned_input_mtime_ns':newest_input.stat().st_mtime_ns,
        'build_output':str(combined),
        'build_mtime_ns':combined.stat().st_mtime_ns,
        'repair_output':str(members),
        'repair_mtime_ns':members.stat().st_mtime_ns,
        'original_outputs_untouched':True,
    }
    (variant/'stage4_rebuild_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('[DONE] Rebuilt and re-evaluated the cleaned-text variant from the verified score shards.')
    print(f'[SAVED] {stage4.OUT / "stage4_variant_summary.csv"}')


if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'[ERROR] {e}',file=sys.stderr)
        sys.exit(1)
