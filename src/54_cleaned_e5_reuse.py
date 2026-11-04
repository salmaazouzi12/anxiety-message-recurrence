#!/usr/bin/env python3
"""Full-corpus cleaned E5 candidate retrieval while reusing unaffected baseline vectors.

This DOES NOT alter src/32_e5_faiss_candidates.py, the baseline index, baseline
embeddings, or the completed cleaned MinHash results.

Run from the project root:
  python src/54_cleaned_e5_reuse.py --stage check
  python src/54_cleaned_e5_reuse.py --stage embed
  python src/54_cleaned_e5_reuse.py --stage retrieve
  python src/53_full_cleaned_candidate_retrieval.py --stage compare

'embed' is resumable at 512 changed texts per checkpoint. 'retrieve' uses the
original IVF-Flat/FAISS settings and is resumable at 2,000 source IDs per shard.
The cleaned text definition is the already-approved Script-38 mapping, NOT a new
cleaning implementation. The E5 representation is 'passage: ' + cleaned text,
exactly as in the original Script 32.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs' / 'anxiety_lineages'
EXP = BASE / 'reviewer_experiments' / 'full_cleaned_retrieval'
OUT = EXP / 'e5_retrieval'
RAW_MAP = BASE / 'minhash' / 'text_id_mapping.parquet'
CLEAN_MAP = BASE / 'stage4_full_ablation' / 'deboilerplated' / 'text_id_mapping_clean.parquet'
CLEAN_LSH_MAP = EXP / 'minhash' / 'text_id_mapping.parquet'
BASE_EMBED = BASE / 'e5_retrieval' / 'e5_embeddings.npy'
BASE_PROGRESS = BASE / 'e5_retrieval' / 'embedding_progress.json'
SOURCE32 = ROOT / 'src' / '32_e5_faiss_candidates.py'
EXP_MANIFEST = EXP / 'input_manifest.json'

CLEAN_EMBED = OUT / 'e5_embeddings.npy'
CLEAN_INDEX = OUT / 'e5_faiss.index'
REUSE_MANIFEST = OUT / 'e5_reuse_manifest.json'
CHECKPOINT_DIR = OUT / 'embedding_reuse_checkpoints'
SUMMARY_FILE = OUT / 'retrieval_summary.json'

EXPECTED = {
    'MODEL_NAME': 'intfloat/multilingual-e5-small',
    'DIM': 384, 'EMBED_BATCH': 64, 'CHECKPOINT_EVERY': 512,
    'QUERY_BATCH': 2000, 'NEIGHBORS': 15,
    'NLIST': 512, 'NPROBE': 32, 'TRAIN_SIZE': 50000, 'SEED': 42,
}


def ensure(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_original_code() -> None:
    ensure(SOURCE32.is_file(), f'Missing original Script 32: {SOURCE32}')
    tree = ast.parse(SOURCE32.read_text(encoding='utf-8-sig'))
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in EXPECTED:
                constants[target.id] = ast.literal_eval(statement.value)
    for key, expected in EXPECTED.items():
        ensure(constants.get(key) == expected,
               f'Script 32 changed {key} from {expected!r} to {constants.get(key)!r}. '
               'STOP: retrieval settings require review.')
    source = SOURCE32.read_text(encoding='utf-8-sig')
    ensure('"passage: " + t' in source,
           'Original Script 32 passage prefix changed; refusing to mix embeddings')
    ensure('normalize_embeddings=True' in source,
           'Original Script 32 embedding normalization changed')
    print('[CHECK] Original Script 32 E5 model, passage prefix and FAISS parameters: MATCH', flush=True)


def read_inputs():
    for p in (RAW_MAP, CLEAN_MAP, CLEAN_LSH_MAP, BASE_EMBED, BASE_PROGRESS, EXP_MANIFEST):
        ensure(p.is_file(), f'Missing required existing file: {p}')
    raw = pd.read_parquet(RAW_MAP)
    clean = pd.read_parquet(CLEAN_MAP)
    lsh = pd.read_parquet(CLEAN_LSH_MAP)
    required = ['text_id', 'exact_family_id', 'normalized_text', 'post_count']
    for name, frame in [('original', raw), ('clean', clean), ('clean MinHash', lsh)]:
        ensure(set(required) <= set(frame.columns), f'{name} mapping missing required columns')
        ensure(len(frame) > 0, f'{name} mapping is empty')
    raw = raw.sort_values('text_id').reset_index(drop=True)
    clean = clean.sort_values('text_id').reset_index(drop=True)
    lsh = lsh.sort_values('text_id').reset_index(drop=True)
    n = len(raw)
    ensure(len(clean) == n and len(lsh) == n, 'Mapping row count mismatch')
    for name, frame in [('original', raw), ('clean', clean), ('clean MinHash', lsh)]:
        ensure(np.array_equal(frame['text_id'].to_numpy(), np.arange(n)),
               f'{name} IDs are not contiguous and ordered')
        ensure(frame['exact_family_id'].is_unique, f'{name} has duplicate exact_family_id')
    ensure(raw['exact_family_id'].equals(clean['exact_family_id']),
           'Cleaned exact family IDs differ from baseline; cannot reuse row-indexed vectors')
    ensure(clean[required].equals(lsh[required]),
           'Cleaned MinHash mapping differs from approved Script-38 cleaned text mapping')
    ensure(raw['post_count'].equals(clean['post_count']), 'Post counts changed')

    # Verify original embedding cache belongs to original, ordered mapping.
    progress = json.loads(BASE_PROGRESS.read_text(encoding='utf-8'))
    fp = hashlib.sha256('\n'.join(raw['exact_family_id'].astype(str)).encode('utf-8')).hexdigest()
    ensure(progress.get('fingerprint') == fp,
           'Original Script 32 embedding cache fingerprint does not match baseline IDs')
    ensure(progress.get('model') == EXPECTED['MODEL_NAME'],
           f"Baseline E5 cache uses a different model: {progress.get('model')}")
    ensure(int(progress.get('completed', -1)) == n,
           f"Original E5 embedding cache not complete ({progress.get('completed')}/{n}); can't reuse yet")
    vectors = np.load(BASE_EMBED, mmap_mode='r', allow_pickle=False)
    ensure(vectors.shape == (n, EXPECTED['DIM']) and vectors.dtype == np.float32,
           f'Unexpected original E5 embeddings shape/dtype: {vectors.shape}, {vectors.dtype}')
    sample = np.linspace(0, n - 1, min(1000, n), dtype=np.int64)
    ensure(np.isfinite(vectors[sample]).all(), 'Original E5 cache contains nonfinite sampled vectors')
    lengths = np.linalg.norm(np.asarray(vectors[sample]), axis=1)
    ensure(np.all((lengths > .97) & (lengths < 1.03)),
           'Original E5 sampled vectors are not unit-normalized')

    original_texts = raw['normalized_text'].fillna('').astype(str)
    cleaned_texts = clean['normalized_text'].fillna('').astype(str)
    changed = np.flatnonzero((original_texts != cleaned_texts).to_numpy()).astype(np.int64)
    print(f'[INPUT] {n:,} text IDs; {len(changed):,} changed; '
          f'{n-len(changed):,} baseline embeddings reusable', flush=True)
    print(f'[INPUT] Original cache complete and aligned with mapping: {BASE_EMBED}', flush=True)

    exp = json.loads(EXP_MANIFEST.read_text(encoding='utf-8'))
    ensure(exp.get('original_mapping_sha256') == sha256_file(RAW_MAP) and
           exp.get('cleaned_mapping_sha256') == sha256_file(CLEAN_MAP),
           'Existing cleaned retrieval experiment manifest mismatches source mappings')
    manifest = {
        'purpose': 'reuse baseline E5 unchanged rows, encode cleaned modified rows, rebuild IVF-Flat',
        'n': n, 'changed': len(changed),
        'original_mapping_sha256': exp['original_mapping_sha256'],
        'cleaned_mapping_sha256': exp['cleaned_mapping_sha256'],
        'original_e5_progress_sha256': sha256_file(BASE_PROGRESS),
        'original_embeddings_size': BASE_EMBED.stat().st_size,
        'original_embeddings_mtime_ns': BASE_EMBED.stat().st_mtime_ns,
        'model': EXPECTED['MODEL_NAME'],
        'original_script32_sha256': sha256_file(SOURCE32),
        'settings': EXPECTED,
    }
    return cleaned_texts.tolist(), changed, vectors, manifest


def replace_with_retry(source: Path, destination: Path) -> None:
    """Commit a fully written temp artifact; retry transient Windows file locks."""
    for attempt in range(15):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 14:
                raise RuntimeError(
                    f'Windows blocked file commit to {destination}. '
                    f'The complete temporary file remains at {source}; do not delete the outputs.'
                )
            time.sleep(min(.2 * (attempt + 1), 1.5))


def verify_e5_model_alignment(model, texts, changed, baseline) -> None:
    """Guard against model/tokenizer drift before mixing old and new embeddings."""
    usable = np.ones(len(texts), dtype=bool)
    usable[changed] = False
    ids = np.flatnonzero(usable)
    ensure(len(ids) > 0,
           'All texts changed: original embedding reuse is unavailable; stop for review')
    probe_ids = ids[np.linspace(0, len(ids)-1, min(12, len(ids)), dtype=np.int64)]
    candidates = model.encode(
        ['passage: ' + texts[int(i)] for i in probe_ids],
        batch_size=EXPECTED['EMBED_BATCH'], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    ensure(candidates.shape == (len(probe_ids), EXPECTED['DIM']),
           'E5 alignment probe returned incorrect vector dimensions')
    similarities = np.sum(candidates * np.asarray(baseline[probe_ids]), axis=1)
    ensure(np.isfinite(similarities).all(), 'E5 alignment probe produced nonfinite similarities')
    ensure(np.all(similarities >= 0.999),
           f'Current E5 model is not aligned with saved original embeddings; '
           f'minimum probe cosine={float(similarities.min()):.6f}. '
           'DO NOT MIX original and newly encoded vectors. '
           'Check local model weights, tokenizer, and SentenceTransformer version.')
    print(f'[VERIFY] Current E5 model matches {len(probe_ids)} sampled unchanged original embeddings '
          f'(minimum cosine={float(similarities.min()):.6f})', flush=True)


def append_checkpoint(completed: int, n_changed: int) -> None:
    """Write immutable checkpoint, avoiding Windows os.replace-on-JSON failures."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINT_DIR / f'{completed:09d}_{uuid.uuid4().hex}.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump({'completed': completed, 'n_changed': n_changed}, f)
        f.flush()
        os.fsync(f.fileno())


def read_checkpoint(n_changed: int) -> int | None:
    if not CHECKPOINT_DIR.exists():
        return None
    checkpoints = []
    for p in CHECKPOINT_DIR.glob('*.json'):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
            count = int(obj['completed'])
            if int(obj['n_changed']) == n_changed and 0 <= count <= n_changed:
                checkpoints.append(count)
        except (ValueError, KeyError, OSError, TypeError, UnicodeError):
            continue
    ensure(bool(checkpoints), f'Checkpoint directory exists but has no valid record: {CHECKPOINT_DIR}')
    return max(checkpoints)


def prepare_output_cache(manifest, n_changed):
    OUT.mkdir(parents=True, exist_ok=True)
    if REUSE_MANIFEST.exists():
        saved = json.loads(REUSE_MANIFEST.read_text(encoding='utf-8'))
        ensure(saved == manifest,
               'E5 reuse manifest differs from current mappings or baseline embeddings. '
               'Refusing to mix partial work. No files overwritten.')
    else:
        existing = [p for p in OUT.iterdir() if p.name not in {'e5_reuse_manifest.json'}]
        ensure(not existing,
               f'Existing E5 outputs without this script\'s manifest: {[p.name for p in existing][:8]}. '
               'Refusing to overwrite or mix another run.')
        with REUSE_MANIFEST.open('x', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    if not CLEAN_EMBED.exists():
        ensure(read_checkpoint(n_changed) is None,
               'Checkpoint exists but cleaned E5 embedding cache is missing')
        temporary = CLEAN_EMBED.with_name('e5_embeddings.partial.npy')
        if temporary.exists():
            temporary.unlink()  # An uncommitted initial copy, never a completed checkpoint.
        print('[COPY] Copying baseline E5 cache to isolated cleaned output (~390 MB)', flush=True)
        shutil.copyfile(BASE_EMBED, temporary)
        replace_with_retry(temporary, CLEAN_EMBED)  # Only this experiment's new file.
        append_checkpoint(0, n_changed)
    elif read_checkpoint(n_changed) is None:
        # Crash between the initial atomic baseline copy and the first checkpoint.
        # No modified row is acknowledged, so the baseline copy is safe to repeat.
        print('[RECOVER] No embedding checkpoint; restoring initial baseline copy', flush=True)
        temporary = CLEAN_EMBED.with_name('e5_embeddings.partial.npy')
        shutil.copyfile(BASE_EMBED, temporary)
        replace_with_retry(temporary, CLEAN_EMBED)
        append_checkpoint(0, n_changed)
    vectors = np.lib.format.open_memmap(CLEAN_EMBED, mode='r+')
    ensure(vectors.shape == (manifest['n'], EXPECTED['DIM']) and vectors.dtype == np.float32,
           'Cleaned E5 cache has unexpected shape or dtype')
    return vectors


def stage_check():
    validate_original_code()
    texts, changed, _base, manifest = read_inputs()
    if REUSE_MANIFEST.exists():
        saved = json.loads(REUSE_MANIFEST.read_text(encoding='utf-8'))
        ensure(saved == manifest, 'Existing E5 experiment manifest does not match; stop')
        print('[CHECK] Existing E5 reuse manifest: MATCH', flush=True)
    else:
        print('[CHECK] No cleaned E5 reuse output created yet', flush=True)
    print('[CHECK] Baseline E5 ready; cleaned IDs match; no files changed', flush=True)
    return len(texts), len(changed)


def stage_embed(device: str):
    validate_original_code()
    texts, changed, base, manifest = read_inputs()
    ensure(not SUMMARY_FILE.exists(),
           'Retrieval already completed; refusing to modify embedding cache afterward')
    ensure(not CLEAN_INDEX.exists(),
           'FAISS index already exists; refusing to modify embeddings behind it')
    cached = prepare_output_cache(manifest, len(changed))
    done = read_checkpoint(len(changed))
    assert done is not None
    print(f'[EMBED] {done:,}/{len(changed):,} changed text embeddings completed; device={device}', flush=True)
    if done == len(changed):
        print('[EMBED] Already complete; no model reloaded', flush=True)
        validate_embedding_reuse(cached, base, changed)
        return

    # Import model ONLY when new vectors actually need encoding.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EXPECTED['MODEL_NAME'], device=device)
    verify_e5_model_alignment(model, texts, changed, base)
    started = time.perf_counter()
    for start in range(done, len(changed), EXPECTED['CHECKPOINT_EVERY']):
        end = min(start + EXPECTED['CHECKPOINT_EVERY'], len(changed))
        ids = changed[start:end]
        passages = ['passage: ' + texts[int(i)] for i in ids]
        vec = model.encode(
            passages, batch_size=EXPECTED['EMBED_BATCH'],
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        ensure(vec.shape == (len(ids), EXPECTED['DIM']),
               f'Unexpected E5 batch shape {vec.shape}')
        ensure(np.isfinite(vec).all(), f'E5 batch at changed row {start} has nonfinite values')
        cached[ids] = vec
        cached.flush()
        append_checkpoint(end, len(changed))
        if end % 10240 < EXPECTED['CHECKPOINT_EVERY'] or end == len(changed):
            elapsed = (time.perf_counter() - started) / 60
            print(f'[EMBED] {end:,}/{len(changed):,} changed rows | session {elapsed:.1f} min',
                  flush=True)
    validate_embedding_reuse(cached, base, changed)
    print('[EMBED] COMPLETE; unchanged rows reused, modified rows encoded and checkpointed', flush=True)


def validate_embedding_reuse(cached, base, changed):
    n = len(base)
    unchanged = np.ones(n, dtype=bool)
    unchanged[changed] = False
    unchanged_ids = np.flatnonzero(unchanged)
    for left in range(0, len(unchanged_ids), 16384):
        ids = unchanged_ids[left:left+16384]
        ensure(np.array_equal(cached[ids], base[ids]),
               'One or more unchanged embeddings differ from baseline!')
    sample = np.linspace(0, n-1, min(3000, n), dtype=np.int64)
    ensure(np.isfinite(cached[sample]).all(), 'Nonfinite values in sampled cleaned E5 rows')
    norms = np.linalg.norm(np.asarray(cached[sample]), axis=1)
    ensure(np.all((norms > .97) & (norms < 1.03)),
           'Cleaned E5 rows not normalized; cache may be corrupt')
    print(f'[VERIFY] {len(unchanged_ids):,} unchanged rows byte-equivalent to baseline', flush=True)


def write_index_atomic(index, destination: Path, faiss):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name('e5_faiss.partial.index')
    faiss.write_index(index, str(tmp))
    ensure(tmp.exists() and tmp.stat().st_size > 0, 'FAISS temporary index write failed')
    replace_with_retry(tmp, destination)


def build_or_load_index(embeddings, n, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    if CLEAN_INDEX.exists():
        print('[FAISS] Loading completed isolated index', flush=True)
        index = faiss.read_index(str(CLEAN_INDEX))
    else:
        print('[FAISS] Training isolated IVF-Flat index (same parameters as Script 32)', flush=True)
        quantizer = faiss.IndexFlatIP(EXPECTED['DIM'])
        index = faiss.IndexIVFFlat(quantizer, EXPECTED['DIM'], EXPECTED['NLIST'], faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(EXPECTED['SEED'])
        sample_ids = rng.choice(n, size=min(EXPECTED['TRAIN_SIZE'], n), replace=False)
        sample_vectors = np.ascontiguousarray(embeddings[sample_ids], dtype=np.float32)
        index.train(sample_vectors)
        for left in range(0, n, 10000):
            right = min(left+10000, n)
            index.add(np.ascontiguousarray(embeddings[left:right], dtype=np.float32))
            if right % 50000 < 10000 or right == n:
                print(f'[FAISS] Indexed {right:,}/{n:,}', flush=True)
        write_index_atomic(index, CLEAN_INDEX, faiss)
    ensure(index.ntotal == n and index.d == EXPECTED['DIM'],
           f'FAISS index dimensions mismatch: {index.ntotal=} {index.d=}')
    ensure(index.nlist == EXPECTED['NLIST'], 'FAISS nlist mismatch')
    index.nprobe = EXPECTED['NPROBE']
    print(f'[FAISS] Ready; indexed={index.ntotal:,} nprobe={index.nprobe}', flush=True)
    return index


def validate_query_shard(p: Path, n: int, left: int, right: int):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(p)
    ensure(set(['text_id_1','text_id_2','semantic_cosine']) <= set(pf.schema_arrow.names),
           f'Unexpected saved shard columns: {p}')
    ensure(pf.metadata.num_rows <= (right-left)*EXPECTED['NEIGHBORS'],
           f'Too many saved pairs for shard {p}')
    # Check IDs and finite similarities; rows can include source as either endpoint.
    for batch in pf.iter_batches(batch_size=25000, columns=['text_id_1','text_id_2','semantic_cosine']):
        d = batch.to_pydict()
        a = np.asarray(d['text_id_1'], dtype=np.int64)
        b = np.asarray(d['text_id_2'], dtype=np.int64)
        score = np.asarray(d['semantic_cosine'], dtype=float)
        ensure(bool(((0 <= a)&(a < n)&(0 <= b)&(b < n)&(a != b)).all()) and
               bool(np.isfinite(score).all()),
               f'Invalid saved E5 candidate shard: {p}')


def stage_retrieve(threads: int):
    validate_original_code()
    texts, changed, base, manifest = read_inputs()
    n = len(texts)
    ensure(REUSE_MANIFEST.exists(), 'Run --stage embed first')
    ensure(json.loads(REUSE_MANIFEST.read_text(encoding='utf-8')) == manifest,
           'Cleaned E5 input manifest mismatch')
    ensure(CLEAN_EMBED.exists(), 'Run --stage embed first')
    ensure(read_checkpoint(len(changed)) == len(changed),
           'Changed E5 embedding cache is not fully encoded. Run --stage embed first.')
    embeddings = np.load(CLEAN_EMBED, mmap_mode='r', allow_pickle=False)
    ensure(embeddings.shape == (n, EXPECTED['DIM']) and embeddings.dtype == np.float32,
           'Invalid cleaned E5 embedding cache')
    validate_embedding_reuse(embeddings, base, changed)
    index = build_or_load_index(embeddings, n, threads)
    import pyarrow.parquet as pq
    import pandas as pd

    total_parts = (n + EXPECTED['QUERY_BATCH'] - 1)//EXPECTED['QUERY_BATCH']
    total_rows = 0
    started = time.perf_counter()
    for part in range(total_parts):
        left = part*EXPECTED['QUERY_BATCH']
        right = min(left+EXPECTED['QUERY_BATCH'], n)
        destination = OUT / f'semantic_pairs_{part:04d}.parquet'
        if destination.exists():
            validate_query_shard(destination, n, left, right)
            total_rows += int(pq.ParquetFile(destination).metadata.num_rows)
            continue
        query_vectors = np.ascontiguousarray(embeddings[left:right], dtype=np.float32)
        similarities, neighbors = index.search(query_vectors, EXPECTED['NEIGHBORS'] + 10)
        pairs = []
        for local_id in range(right-left):
            src = left+local_id
            kept = 0
            for score, neighbor in zip(similarities[local_id], neighbors[local_id]):
                j = int(neighbor)
                if j < 0 or j == src:
                    continue
                pairs.append((min(src,j), max(src,j), float(score)))
                kept += 1
                if kept >= EXPECTED['NEIGHBORS']:
                    break
        if pairs:
            frame = pd.DataFrame(pairs, columns=['text_id_1','text_id_2','semantic_cosine'])
            frame = frame.groupby(['text_id_1','text_id_2'], as_index=False)['semantic_cosine'].max()
        else:
            frame = pd.DataFrame(columns=['text_id_1','text_id_2','semantic_cosine'])
        tmp = OUT / f'semantic_pairs_{part:04d}.partial.parquet'
        frame.to_parquet(tmp, index=False)
        replace_with_retry(tmp, destination)
        total_rows += len(frame)
        if (part + 1)%10 == 0 or part+1 == total_parts:
            print(f'[RETRIEVE] {right:,}/{n:,} IDs | saved rows={total_rows:,} | '
                  f'session {(time.perf_counter()-started)/60:.1f} min', flush=True)

    result = {
        'unique_texts': n,
        'model': EXPECTED['MODEL_NAME'],
        'embedding_dimensions': EXPECTED['DIM'],
        'reused_unchanged_embeddings': n-len(changed),
        'reencoded_changed_embeddings': len(changed),
        'faiss_index': 'IVF-Flat',
        'nlist': EXPECTED['NLIST'], 'nprobe': EXPECTED['NPROBE'],
        'neighbors_per_text': EXPECTED['NEIGHBORS'],
        'saved_pair_rows_before_global_dedup': total_rows,
        'candidate_files': total_parts,
        'status': 'semantic_candidate_retrieval_complete',
        'limitations': [
            'Approximate nearest neighbor retrieval',
            'Long texts may be truncated by the encoder',
            'Semantic similarity does not establish textual reuse',
            'Global pair deduplication still required',
        ],
    }
    if SUMMARY_FILE.exists():
        existing = json.loads(SUMMARY_FILE.read_text(encoding='utf-8'))
        ensure(existing == result, 'Existing retrieval summary differs from newly verified shards')
        print('[RETRIEVE] Completed summary already exists; verified all output shards', flush=True)
    else:
        with SUMMARY_FILE.open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
    print('[RESULT]', json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['check','embed','retrieve'])
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu',
                   help='SentenceTransformer device for changed text encoding (default: cpu)')
    p.add_argument('--threads', type=int, default=4,
                   help='FAISS CPU threads (default: 4, same as original Script 32)')
    args = p.parse_args()
    ensure(args.threads >= 1, '--threads must be >= 1')
    if args.stage == 'check':
        stage_check()
    elif args.stage == 'embed':
        stage_embed(args.device)
    elif args.stage == 'retrieve':
        stage_retrieve(args.threads)


if __name__ == '__main__':
    main()
