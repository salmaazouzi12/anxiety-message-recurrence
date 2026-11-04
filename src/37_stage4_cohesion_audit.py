#!/usr/bin/env python3
"""Stage 4, anxiety-research: original-lineage integrity + boilerplate error audit.

IMPORTANT: READ-ONLY. This is NOT a re-run of scripts 34/35 and does not
pretend to recreate their cohesion rules. It verifies original family labels,
quantifies annotated merge/split errors, audits text-template artifacts, and
packages the LOCAL source scripts needed to implement a faithful ablation.

Run from the project root:
    python src/37_stage4_cohesion_audit.py

Requires: pandas, pyarrow (already available if Stage 3 ran).
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Install missing package: python -m pip install pandas")
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # Audit of labels still works; final Parquet integrity check is skipped.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "src" else HERE
SRC = ROOT / "src"
BASE = ROOT / "outputs" / "anxiety_lineages"
OUT = BASE / "stage4_cohesion_audit"
PAIR_NAMES = ("pair_sampling_key.csv", "pair_annotations_completed.csv")

ID_COLS = ["text_id", "text_id_1", "text_id_a", "semantic_id", "exact_text_id", "unique_text_id", "exact_id"]
LINEAGE_COLS = ["lineage_id", "family_id", "repaired_lineage_id", "final_lineage_id", "cluster_id"]
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
HASHTAG = re.compile(r"(?<!\w)#[\w]+", re.UNICODE)
MENTION = re.compile(r"(?<!\w)@[\w.]+", re.UNICODE)
SPACES = re.compile(r"\s+")


def csv_path(name: str) -> Path | None:
    options = [SRC / name, HERE / name, ROOT / name,
               ROOT / "data" / name, ROOT / "outputs" / name,
               ROOT / "outputs" / "human_validation" / name]
    for p in options:
        if p.is_file():
            return p
    # Do not select an arbitrary duplicate from dozens of stale exports.
    extras = list(ROOT.glob("**/" + name))
    return extras[0] if len(extras) == 1 else None


def col(cols, possibilities):
    lookup = {s.lower(): s for s in cols}
    return next((lookup[s.lower()] for s in possibilities if s.lower() in lookup), None)


def canonical_id(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if s in {"", "nan", "None", "<NA>"}:
        return None
    try:
        return str(int(s))
    except (TypeError, ValueError):
        return s


def clean(text: str) -> str:
    text = str(text or "").lower()
    text = URL.sub(" ", text)
    text = HASHTAG.sub(" ", text)
    text = MENTION.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def ratio(a: str, b: str) -> float:
    # Diagnostic only: this is NOT Script 28's lexical_cosine scorer.
    return SequenceMatcher(None, a[:1500], b[:1500], autojunk=True).ratio()


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.95996398454005
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return [round(center-radius, 5), round(center+radius, 5)]


def save_csv(path, rows, headers=None):
    fields = headers or (list(rows[0].keys()) if rows else [])
    if not fields:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_membership_files():
    # Repaired version is the true final output of Script 35; do not silently
    # substitute un-repaired Script 34 membership.
    found = sorted(BASE.rglob("repaired_lineage_members.parquet")) if BASE.is_dir() else []
    if not found:
        found = sorted(BASE.rglob("*repaired*members*.parquet")) if BASE.is_dir() else []
    return found


def inspect_parquet(path):
    schema = pq.ParquetFile(path).schema_arrow.names
    return {"relative_path": str(path.relative_to(ROOT)), "columns": schema,
            "rows": pq.ParquetFile(path).metadata.num_rows}


def lookup_membership(path, ids):
    """Scan original final lineage membership for only sampled IDs."""
    cols = pq.ParquetFile(path).schema_arrow.names
    id_col, line_col = col(cols, ID_COLS), col(cols, LINEAGE_COLS)
    if not id_col or not line_col:
        return {}, {"usable": False, "columns": cols,
                    "message": "Cannot identify ID / lineage columns; need Scripts 34/35."}
    matches = defaultdict(set)
    for batch in pq.ParquetFile(path).iter_batches(columns=[id_col, line_col], batch_size=250_000):
        b = batch.to_pydict()
        for item, group in zip(b[id_col], b[line_col]):
            key = canonical_id(item)
            if key in ids and group is not None and str(group).strip():
                matches[key].add(str(group).strip())
    conflicting = sum(len(v)>1 for v in matches.values())
    return {k: next(iter(v)) for k,v in matches.items() if len(v)==1}, {
        "usable": True, "id_column": id_col, "lineage_column": line_col,
        "sample_ids_found": len(matches), "sample_ids_conflicting": conflicting,
        "sample_ids_missing": len(ids)-len(matches)
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[INFO] Project:", ROOT)
    path_key = csv_path(PAIR_NAMES[0])
    path_labels = csv_path(PAIR_NAMES[1])
    if not path_key or not path_labels:
        sys.exit("[ERROR] Missing pair_sampling_key.csv / pair_annotations_completed.csv. "
                 "Put both inside src/ exactly as in Stage 3.")

    key = pd.read_csv(path_key, dtype={"pair_id": "string", "text_id_a": "string", "text_id_b": "string"})
    labels = pd.read_csv(path_labels, dtype={"pair_id": "string"})
    if key.pair_id.duplicated().any() or labels.pair_id.duplicated().any():
        sys.exit("[ERROR] Duplicate pair IDs in the sample or annotations")
    if set(key.pair_id) != set(labels.pair_id):
        sys.exit("[ERROR] Pair IDs don't match between two annotation files")
    rows = key.merge(labels, on="pair_id", validate="one_to_one")
    if len(rows) != 400:
        print(f"[WARNING] Expected 400 pairs; loaded {len(rows)}. Using actual count.")
    labelset = {"SAME", "RELATED", "UNRELATED"}
    rows["label"] = rows["label"].astype(str).str.strip().str.upper()
    if not rows.label.isin(labelset).all():
        sys.exit("[ERROR] Invalid or missing semantic labels")

    rows["text_id_a"] = rows["text_id_a"].map(canonical_id)
    rows["text_id_b"] = rows["text_id_b"].map(canonical_id)
    if rows[["text_id_a", "text_id_b"]].isna().any().any():
        sys.exit("[ERROR] Missing text IDs in pair sampling key")

    rows["original_prediction"] = (rows.lineage_a.notna() & rows.lineage_b.notna() &
                                    (rows.lineage_a == rows.lineage_b)).astype(int)
    rows["same_label"] = (rows.label == "SAME").astype(int)
    rows["original_correct"] = rows.original_prediction.eq(rows.same_label)
    rows["error_kind"] = [
        "true_positive" if p == 1 and gold == 1 else
        "false_merge" if p == 1 and gold == 0 else
        "missed_link_in_sample" if p == 0 and gold == 1 else
        "true_negative" for p,gold in zip(rows.original_prediction, rows.same_label)
    ]

    # Read only actual repaired membership and compare to historical sampling key.
    paths = find_membership_files()
    member_audit = {"status": "missing_repaired_membership", "files": []}
    if len(paths) == 1 and pq is not None:
        selected = paths[0]
        member_audit = {"status": "found", "path": str(selected.relative_to(ROOT)),
                        "files": [inspect_parquet(selected)]}
        selected_ids = set(rows.text_id_a) | set(rows.text_id_b)
        mapping, scan = lookup_membership(selected, selected_ids)
        member_audit["scan"] = scan
        if scan.get("usable"):
            rows["final_member_a"] = rows.text_id_a.map(mapping)
            rows["final_member_b"] = rows.text_id_b.map(mapping)
            # Missing sampled IDs are not automatically negative; preserve uncertainty.
            complete = rows.final_member_a.notna() & rows.final_member_b.notna()
            final_same = complete & rows.final_member_a.eq(rows.final_member_b)
            rows["final_both_ids_found"] = complete
            rows["final_prediction"] = final_same.astype("Int64").where(complete, pd.NA)
            matched = rows.loc[complete]
            member_audit["comparison"] = {
                "pairs_with_both_ids_found": int(complete.sum()),
                "disagreements_with_sampling_key": int((matched.final_prediction.astype(int) != matched.original_prediction).sum()),
            }
            if member_audit["comparison"]["disagreements_with_sampling_key"]:
                print("[WARNING] Repaired family assignment differs from stored sampling key: check versions!")
    elif len(paths) > 1:
        member_audit = {"status": "ambiguous_membership", "paths": [str(p.relative_to(ROOT)) for p in paths],
                        "note": "Do not automatically select one of multiple final versions."}
    elif len(paths) == 1:
        member_audit = {"status": "pyarrow_not_installed", "path": str(paths[0].relative_to(ROOT))}

    records = []
    for r in rows.to_dict("records"):
        raw_a, raw_b = str(r.get("text_a") or ""), str(r.get("text_b") or "")
        cl_a, cl_b = clean(raw_a), clean(raw_b)
        raw_sim = ratio(raw_a.lower(), raw_b.lower())
        clean_sim = ratio(cl_a, cl_b) if cl_a and cl_b else None
        ra = len(raw_a)
        rb = len(raw_b)
        trimmed = 1 - (len(cl_a)+len(cl_b))/max(ra+rb, 1)
        r["raw_sequence_ratio_diagnostic"] = round(raw_sim, 5)
        r["stripped_sequence_ratio_diagnostic"] = round(clean_sim, 5) if clean_sim is not None else None
        r["similarity_change_after_strip"] = round(clean_sim-raw_sim, 5) if clean_sim is not None else None
        r["removed_character_fraction"] = round(trimmed, 5)
        r["pair_has_url_or_tag"] = bool(URL.search(raw_a+" "+raw_b) or HASHTAG.search(raw_a+" "+raw_b) or MENTION.search(raw_a+" "+raw_b))
        # This flag prioritizes manual review; it does NOT prove boilerplate causality.
        r["boilerplate_review_priority"] = bool(r["error_kind"] == "false_merge" and (
            trimmed >= 0.25 or (clean_sim is not None and (clean_sim-raw_sim) <= -0.10)))
        r["text_a"] = raw_a
        r["text_b"] = raw_b
        records.append(r)

    # Small, reproducible numerical summaries. Samples were deliberately stratified;
    # group-level rates must not be interpreted as population estimates.
    overall = Counter(r["error_kind"] for r in records)
    group_stats = []
    for group_name in ("source_group", "sampling_stratum", "annotation_origin"):
        for group_val in sorted(set(str(r.get(group_name) or "") for r in records)):
            subset = [r for r in records if str(r.get(group_name) or "") == group_val]
            c = Counter(r["error_kind"] for r in subset)
            tp, fp, fn, tn = [c[k] for k in ("true_positive", "false_merge", "missed_link_in_sample", "true_negative")]
            positive = tp + fp
            gold = tp + fn
            group_stats.append({
                "group_field": group_name, "group_value": group_val, "sample_n": len(subset),
                "tp": tp, "false_merges": fp, "missed_links": fn, "tn": tn,
                "precision_conditional": round(tp/positive, 5) if positive else None,
                "precision_ci_low": wilson(tp, positive)[0],
                "precision_ci_high": wilson(tp, positive)[1],
                "recall_conditional": round(tp/gold, 5) if gold else None,
                "boilerplate_priority_cases": sum(bool(r["boilerplate_review_priority"]) for r in subset),
            })

    # Include ORIGINAL approved annotations but not AI agreement as gold standard.
    save_csv(OUT/"stage4_pair_audit.csv", records)
    save_csv(OUT/"stage4_group_metrics.csv", group_stats)
    focus = [r for r in records if r["error_kind"] in ("false_merge", "missed_link_in_sample")]
    focus.sort(key=lambda r: (not bool(r["boilerplate_review_priority"]), r["error_kind"], r["pair_id"]))
    save_csv(OUT/"stage4_error_review.csv", focus)
    counts = dict(overall)
    files = {}
    for kind, pattern in [
        ("lexical_scores", "lexical_scoring/scored_pairs_*.parquet"),
        ("novel_semantic_scores", "combined_retrieval/novel_e5_scored/**/*.parquet"),
        ("repaired_families", "combined_cohesion_repair/*"),
        ("cohesion_audit", "**/cohesion_audit.parquet")]:
        paths2 = sorted(BASE.glob(pattern)) if BASE.is_dir() else []
        files[kind] = {"count": len(paths2), "examples": [str(x.relative_to(ROOT)) for x in paths2[:5]]}
    code_paths = []
    for nr in ("27", "28", "29", "31", "32", "33", "34", "35"):
        code_paths.extend(sorted(SRC.glob(f"{nr}_*.py")))
    code_paths = sorted(set(code_paths))
    expected = {"34": [p.name for p in code_paths if p.name.startswith("34_")],
                "35": [p.name for p in code_paths if p.name.startswith("35_")]}
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "project_root": str(ROOT), "n_labeled_pairs": len(records),
                "confusion_matrix": counts,
                "boilerplate_priority_false_merges": sum(bool(r["boilerplate_review_priority"]) for r in records),
                "membership_audit": member_audit,
                "candidate_files": files, "source_scripts": [p.name for p in code_paths],
                "main_scripts_found": expected,
                "limitations": [
                    "No reconstruction or full-cohesion ablation has been run.",
                    "SequenceMatcher ratios are diagnostics, NOT the Script 28 HashingVectorizer lexical_cosine.",
                    "A deliberately stratified sample cannot give population recall or unbiased overall precision.",
                    "Approved labels include 53 originally AI-proposed, human-accepted judgments.",
                    "A split/missed link is evidenced only among sampled candidate pairs, not all corpus pairs."]}
    (OUT/"stage4_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # This ZIP is convenient for requesting a faithful reconstruction edit;
    # only Python source & manifest are copied. No Parquet, post text or annotations.
    bundle = OUT/"stage4_source_bundle.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/"stage4_manifest.json", arcname="stage4_manifest.json")
        for src in code_paths:
            z.write(src, arcname="src/" + src.name)
    readme = f"""STAGE 4 READ-ONLY AUDIT — anxiety-research

Annotated pairs loaded: {len(records)}
Original assignment TP={counts.get('true_positive',0)}, false merges={counts.get('false_merge',0)},
missed sampled links={counts.get('missed_link_in_sample',0)}, TN={counts.get('true_negative',0)}.
Potential boilerplate-driven false merges requiring review: {manifest['boilerplate_priority_false_merges']}.

Repaired final membership: {member_audit.get('status')}.
Local originals Scripts 34/35: {expected}.

FILES:
- stage4_pair_audit.csv: row-level original decision + reference labels + template diagnostics
- stage4_group_metrics.csv: stratified performance and confidence intervals
- stage4_error_review.csv: all original false merges and sampled missed links
- stage4_manifest.json: discovered original inputs, schemas and integrity checks
- stage4_source_bundle.zip: source Scripts 27-35, if available; NO post text

IMPORTANT:
This is a diagnostic and data-integrity audit. It DOES NOT rerun graph construction,
boilerplate-stripped candidate retrieval or cohesion repair. For a publishable
Stage 4 ablation, modify and run actual original scripts 34/35 (and possibly 28/33),
then recompute recurrence coverage, cross-page share and temporal statistics.
Do not report the diagnostics as corpus-wide precision/recall or causal evidence.
"""
    (OUT/"README_STAGE4.txt").write_text(readme,encoding="utf-8")
    print(f"[DONE] TP={counts.get('true_positive',0)}, false merges={counts.get('false_merge',0)}, "
          f"sampled missed links={counts.get('missed_link_in_sample',0)}, TN={counts.get('true_negative',0)}")
    print(f"[INFO] Possible boilerplate-priority false merges: {manifest['boilerplate_priority_false_merges']}")
    print(f"[INFO] Scripts 34/35 present: {bool(expected['34'])}/{bool(expected['35'])}")
    print(f"[SAVED] {OUT / 'stage4_error_review.csv'}")
    print(f"[SAVED] {bundle}")
    print("[NOTE] Read README_STAGE4.txt. This audit does NOT constitute full reconstruction ablation.")


if __name__ == "__main__":
    main()
