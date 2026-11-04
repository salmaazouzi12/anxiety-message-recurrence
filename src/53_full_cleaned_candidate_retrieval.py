#!/usr/bin/env python3
"""Task 4: faithful, isolated full-corpus candidate retrieval on the EXISTING
combined-cleaning text map (not a new cleaning recipe).

Run from the anxiety-research project root:
    python src/53_full_cleaned_candidate_retrieval.py --stage check
    python src/53_full_cleaned_candidate_retrieval.py --stage minhash
    python src/53_full_cleaned_candidate_retrieval.py --stage e5
    python src/53_full_cleaned_candidate_retrieval.py --stage compare

Source implementations: src/27_minhash_candidates.py and
src/32_e5_faiss_candidates.py. These source files are NOT changed. Instead,
verified path substitutions produce working copies in an isolated output folder.
The existing cleaned mapping comes from 38_stage4_faithful_ablation.py:
outputs/anxiety_lineages/stage4_full_ablation/deboilerplated/text_id_mapping_clean.parquet

Optimized MinHash stage uses the SAME exact MinHash/LSH algorithm but updates
signatures in batches with optional parallel workers and checkpoints both phases.
Restart only if migrating from an incomplete OLD runner with incompatible shards.
Script 32 has resumable embedding and query shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "anxiety_lineages"
EXP = BASE / "reviewer_experiments" / "full_cleaned_retrieval"
RAW = BASE / "minhash" / "text_id_mapping.parquet"
CLEAN = BASE / "stage4_full_ablation" / "deboilerplated" / "text_id_mapping_clean.parquet"
SOURCE27 = ROOT / "src" / "27_minhash_candidates.py"
SOURCE32 = ROOT / "src" / "32_e5_faiss_candidates.py"
LEX = EXP / "minhash"
SEM = EXP / "e5_retrieval"
PREPARED = EXP / "prepared_source.parquet"
MANIFEST = EXP / "input_manifest.json"
GEN = EXP / "generated"


def require(cond, why):
    if not cond:
        raise RuntimeError(why)


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    require(count == 1,
            f"{name}: expected exact original code fragment once; found {count}: {old[:85]!r}. "
            "STOP: original script differs; no unsafe substitution will be made.")
    return source.replace(old, new, 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def make_derived_source(which: str) -> str:
    if which == 'minhash':
        src = SOURCE27.read_text(encoding='utf-8-sig')
        src = replace_once(src, 'ROOT = Path(__file__).resolve().parents[1]',
                           'ROOT = Path(__file__).resolve().parents[5]', which)
        src = replace_once(src, 'SOURCE = BASE / "unique_text_families.parquet"',
                           'SOURCE = BASE / "reviewer_experiments" / "full_cleaned_retrieval" / "prepared_source.parquet"', which)
        src = replace_once(src, 'OUT = BASE / "minhash"',
                           'OUT = BASE / "reviewer_experiments" / "full_cleaned_retrieval" / "minhash"', which)
    elif which == 'e5':
        src = SOURCE32.read_text(encoding='utf-8-sig')
        src = replace_once(src, 'ROOT = Path(__file__).resolve().parents[1]',
                           'ROOT = Path(__file__).resolve().parents[5]', which)
        src = replace_once(src, 'SOURCE = BASE / "minhash" / "text_id_mapping.parquet"',
                           'SOURCE = BASE / "reviewer_experiments" / "full_cleaned_retrieval" / "minhash" / "text_id_mapping.parquet"', which)
        src = replace_once(src, 'OUT = BASE / "e5_retrieval"',
                           'OUT = BASE / "reviewer_experiments" / "full_cleaned_retrieval" / "e5_retrieval"', which)
        old = '''fingerprint = hashlib.sha256(
    "\\n".join(
        mapping["exact_family_id"].astype(str)
    ).encode("utf-8")
).hexdigest()'''
        new = '''# Fingerprint cleaned content AS WELL AS IDs for safe embedding-cache resume.
_fingerprint_hasher = hashlib.sha256()
for _ident, _content in zip(mapping["exact_family_id"].astype(str), texts):
    _fingerprint_hasher.update(_ident.encode("utf-8"))
    _fingerprint_hasher.update(b"\\x00")
    _fingerprint_hasher.update(_content.encode("utf-8"))
    _fingerprint_hasher.update(b"\\x00")
fingerprint = _fingerprint_hasher.hexdigest()'''
        src = replace_once(src, old, new, 'e5 content-aware fingerprint')
    else:
        raise ValueError(which)
    # Check generated code syntax BEFORE writing or running it.
    compile(src, f'<derived {which}>', 'exec')
    return src


def read_input_mappings():
    import pandas as pd
    import numpy as np

    for path in [SOURCE27, SOURCE32, RAW, CLEAN]:
        require(path.exists(), f"Missing required EXISTING file: {path}")
    original = pd.read_parquet(RAW)
    cleaned = pd.read_parquet(CLEAN)
    required = {'text_id', 'exact_family_id', 'normalized_text', 'post_count'}
    for name, df in [('baseline', original), ('cleaned', cleaned)]:
        require(required.issubset(df.columns), f"{name} mapping lacks {required-set(df.columns)}")
    original = original.sort_values('text_id').reset_index(drop=True)
    cleaned = cleaned.sort_values('text_id').reset_index(drop=True)
    require(len(original) == len(cleaned), 'Cleaned/baseline row counts differ')
    n = len(original)
    ids = np.arange(n, dtype=np.int64)
    require(np.array_equal(original.text_id.to_numpy(), ids), 'Original IDs not contiguous, zero-based')
    require(np.array_equal(cleaned.text_id.to_numpy(), ids), 'Cleaned IDs not contiguous, zero-based')
    require(original.exact_family_id.equals(cleaned.exact_family_id),
            'Cleaned exact-family IDs differ from original: pair identity NOT preserved')
    require(original.post_count.equals(cleaned.post_count),
            'Cleaned post_count differs from original')
    require(cleaned.exact_family_id.is_unique, 'Cleaned mapping has duplicate exact_family_id')
    altered = (original.normalized_text.fillna('').astype(str) !=
               cleaned.normalized_text.fillna('').astype(str)).sum()
    print(f'[INPUT] {n:,} identical text IDs; {altered:,} texts modified by established combined cleaning')
    print(f'[INPUT] Original map: {RAW.relative_to(ROOT)}')
    print(f'[INPUT] Cleaned map: {CLEAN.relative_to(ROOT)}')
    return cleaned, n, int(altered)


def current_manifest(n, altered):
    return {
        'experiment': 'full_cleaned_candidate_retrieval',
        'cleaning_definition': 'existing deboilerplated mapping from Script 38',
        'text_ids': n,
        'modified_texts': altered,
        'original_mapping_sha256': sha256_file(RAW),
        'cleaned_mapping_sha256': sha256_file(CLEAN),
        'source27_sha256': sha256_file(SOURCE27),
        'source32_sha256': sha256_file(SOURCE32),
    }


def guard_manifest(expected, create=False):
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding='utf-8'))
        require(existing == expected,
                f"Experiment inputs changed since a prior run. Refusing to mix results. "
                f"Inspect {MANIFEST} and experiment outputs; no files changed.")
    elif create:
        EXP.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(expected, indent=2), encoding='utf-8')
    else:
        print('[NOTE] Experiment manifest not written yet. Run --stage minhash first.')


def validate_minhash_mapping(cleaned):
    import pandas as pd
    saved = LEX / 'text_id_mapping.parquet'
    require(saved.exists(), 'Cleaned MinHash mapping absent; run --stage minhash first')
    d = pd.read_parquet(saved)
    d = d.sort_values('text_id').reset_index(drop=True)
    expected = cleaned[['text_id', 'exact_family_id', 'normalized_text', 'post_count']].reset_index(drop=True)
    require(d[expected.columns].equals(expected),
            'Produced MinHash mapping differs from supplied cleaned mapping!')


def stage_minhash(cleaned, manifest, restart):
    import pandas as pd
    summary = LEX / 'minhash_summary.json'
    if summary.exists() and not restart:
        validate_minhash_mapping(cleaned)
        print('[REUSE] Full cleaned MinHash already finished:', summary)
        print('[RESULT]', summary.read_text(encoding='utf-8'))
        return
    if LEX.exists():
        old = list(LEX.iterdir())
        if old and not restart:
            raise RuntimeError(
                f"Partial lexical results already exist at {LEX} but no completed summary. "
                'Original Script 27 cannot resume its LSH index. Run again using '
                '--stage minhash --restart-minhash to delete ONLY partial files in this experiment.')
        if restart:
            # Only this isolated directory is affected; E5 results must not be mixed with a new candidate pool.
            require(not SEM.exists() or not list(SEM.iterdir()),
                    'Semantic outputs already exist: do not restart lexical retrieval under them.')
            for p in old:
                require(p.is_file() and (p.name.startswith('candidate_pairs_') or
                                       p.name in {'text_id_mapping.parquet', 'minhash_summary.json'}),
                        f'Refusing to delete an unexpected file in isolated output: {p}')
                p.unlink()
    EXP.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    LEX.mkdir(parents=True, exist_ok=True)
    # Source 27 internally generates text_id via reset_index. Drop the incoming text_id
    # without changing row order, so it will assign exactly the original IDs.
    cleaned[['exact_family_id', 'normalized_text', 'post_count']].to_parquet(PREPARED, index=False)
    guard_manifest(manifest, create=True)
    generated = GEN / '27_cleaned_full_retrieval.py'
    generated.write_text(make_derived_source('minhash'), encoding='utf-8')
    print('[RUN] Faithful original 27 algorithm, ALL text IDs; isolated output', flush=True)
    subprocess.run([sys.executable, str(generated)], cwd=ROOT, check=True)
    require(summary.exists(), 'Source 27 exited without its summary')
    validate_minhash_mapping(cleaned)
    print('[PASS] Rebuilt MinHash mapping exactly preserves the cleaned original text IDs')
    print('[RESULT]', summary.read_text(encoding='utf-8'))


def stage_e5(cleaned, manifest):
    guard_manifest(manifest, create=False)
    require(MANIFEST.exists(), 'Run --stage minhash first (no experiment manifest found)')
    require((LEX / 'minhash_summary.json').exists(), 'Full cleaned MinHash not completed')
    validate_minhash_mapping(cleaned)
    generated = GEN / '32_cleaned_full_retrieval.py'
    GEN.mkdir(parents=True, exist_ok=True)
    generated.write_text(make_derived_source('e5'), encoding='utf-8')
    print('[RUN] Faithful original 32 E5 + FAISS, ALL text IDs; isolated, resumable cache', flush=True)
    subprocess.run([sys.executable, str(generated)], cwd=ROOT, check=True)
    summary = SEM / 'retrieval_summary.json'
    require(summary.exists(), 'Source 32 exited without its summary')
    print('[RESULT]', summary.read_text(encoding='utf-8'))


def pair_files(folder, prefix):
    files = sorted(folder.glob(prefix))
    require(files, f'No candidate parquet shards found: {folder / prefix}')
    return files


def stage_compare(n, manifest, lexical_only):
    import pyarrow.parquet as pq
    import numpy as np

    guard_manifest(manifest, create=False)
    require(MANIFEST.exists(), 'Run --stage minhash first')
    require((LEX / 'minhash_summary.json').exists(), 'Full cleaned MinHash missing')
    sources = [
        ('original_minhash', 1, pair_files(BASE / 'minhash', 'candidate_pairs_*.parquet')),
        ('cleaned_minhash', 2, pair_files(LEX, 'candidate_pairs_*.parquet')),
    ]
    if not lexical_only:
        require((SEM / 'retrieval_summary.json').exists(),
                'Full cleaned E5 not finished; run --stage e5 or --stage compare --lexical-only')
        sources += [
            ('original_e5', 4, pair_files(BASE / 'e5_retrieval', 'semantic_pairs_*.parquet')),
            ('cleaned_e5', 8, pair_files(SEM, 'semantic_pairs_*.parquet')),
        ]
    EXP.mkdir(parents=True, exist_ok=True)
    db_path = EXP / 'pair_comparison.sqlite'
    if db_path.exists():
        db_path.unlink()  # This experiment's derived comparison database ONLY.
    con = sqlite3.connect(str(db_path))
    try:
        con.execute('PRAGMA journal_mode=OFF')
        con.execute('PRAGMA synchronous=OFF')
        con.execute('CREATE TABLE pairs (pair_key INTEGER PRIMARY KEY, flags INTEGER NOT NULL)')
        for label, flag, files in sources:
            print(f'[COMPARE] {label}: {len(files)} candidate shards', flush=True)
            rows_seen = 0
            for fi, path in enumerate(files):
                pf = pq.ParquetFile(path)
                require({'text_id_1', 'text_id_2'}.issubset(pf.schema_arrow.names),
                        f'Unexpected pair columns in {path}')
                for batch in pf.iter_batches(batch_size=50000, columns=['text_id_1', 'text_id_2']):
                    c = batch.to_pydict()
                    a = np.asarray(c['text_id_1'], dtype=np.int64)
                    b = np.asarray(c['text_id_2'], dtype=np.int64)
                    require(bool(np.all((a >= 0) & (b >= 0) & (a < n) & (b < n))),
                            f'Out-of-range ID found in {path}')
                    lo = np.minimum(a, b)
                    hi = np.maximum(a, b)
                    keys = (lo[lo != hi] * n + hi[lo != hi]).tolist()
                    con.executemany(
                        'INSERT INTO pairs (pair_key, flags) VALUES (?, ?) '
                        'ON CONFLICT(pair_key) DO UPDATE SET flags=(flags | excluded.flags)',
                        ((int(k), flag) for k in keys),
                    )
                    rows_seen += len(a)
                if (fi + 1) % 20 == 0:
                    con.commit()
                    print(f'  {fi+1}/{len(files)} files; {rows_seen:,} rows read', flush=True)
            con.commit()
            print(f'[COMPARE] {label}: {rows_seen:,} raw rows read', flush=True)
        def counts(old_mask, new_mask):
            row = con.execute('''SELECT
                SUM(CASE WHEN (flags & ?) != 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN (flags & ?) != 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN (flags & ?) != 0 AND (flags & ?) != 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN (flags & ?) = 0 AND (flags & ?) != 0 THEN 1 ELSE 0 END),
                SUM(CASE WHEN (flags & ?) != 0 AND (flags & ?) = 0 THEN 1 ELSE 0 END)
                FROM pairs''', (old_mask, new_mask, old_mask, new_mask,
                               old_mask, new_mask, old_mask, new_mask)).fetchone()
            before, after, shared, new_only, lost = [int(x or 0) for x in row]
            return {'original': before, 'cleaned': after, 'shared': shared,
                    'new_only': new_only, 'lost': lost,
                    'net_difference': after-before,
                    'jaccard': (shared/(before+after-shared)) if before+after-shared else None}
        report = {'n_text_ids': n,
                  'comparison_scope': 'full fixed text-ID universe, direct unfiltered candidate pairs',
                  'minhash': counts(1, 2)}
        if not lexical_only:
            report['e5'] = counts(4, 8)
            report['union_minhash_e5'] = counts(5, 10)
        (EXP / 'candidate_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('[RESULT]', json.dumps(report, indent=2), flush=True)
        print('[SAVED]', EXP / 'candidate_comparison.json', flush=True)
    finally:
        con.close()


# ----- faster, resumable MinHash compatible with original script 27 -----
import os
import re
import time
import uuid
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor

SIGNATURES = EXP / 'minhash_signatures.npy'
SIG_PROGRESS = EXP / 'minhash_signatures_progress.json'
LSH_PROGRESS = EXP / 'minhash_lsh_progress.json'
NUM_PERM = 64
SHINGLE_SIZE = 5
LSH_THRESHOLD = 0.45
SIGNATURE_CHUNK = 512
RETRIEVAL_CHUNK = 2000


def save_json_atomic(path, obj):
    """Write one-time summaries atomically; use unique temps and retry Windows locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    for attempt in range(12):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 11:
                raise RuntimeError(f'Windows is blocking replacement of {path}. '
                                   f'New data remains safe in {temp}.')
            time.sleep(min(0.2 * (attempt + 1), 1.0))


def _checkpoint_folder(path):
    return path.with_name(path.stem + '_checkpoints')


def has_progress(path):
    folder = _checkpoint_folder(path)
    return path.exists() or (folder.is_dir() and any(folder.glob('*.json')))


def load_progress(path):
    """Recover the highest complete durable checkpoint, including legacy JSON."""
    candidates = []
    if path.exists():
        candidates.append(path)
    folder = _checkpoint_folder(path)
    if folder.is_dir():
        candidates.extend(folder.glob('*.json'))
    valid = []
    for file in candidates:
        try:
            obj = json.loads(file.read_text(encoding='utf-8'))
            completed = int(obj['completed'])
            if completed < 0:
                continue
            valid.append((completed, obj, file))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            # A crash can leave an unfinished journal entry; ignore it.
            continue
    if not valid:
        raise RuntimeError(f'Checkpoint files exist but none is valid for {path}')
    completed, obj, file = max(valid, key=lambda x: x[0])
    print(f'[RESUME] {path.name}: recovered {completed:,} entries from {file.name}', flush=True)
    return obj


def save_progress(path, obj):
    """Append-only Windows-safe checkpoints: NEVER replace an existing JSON file.

    The numeric progress is valid only after the associated signature memmap
    flush or retrieval parquet shard write has completed. Each record has a
    unique filename, so antivirus/indexing locks on older JSON are irrelevant.
    """
    folder = _checkpoint_folder(path)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{int(obj['completed']):09d}_{uuid.uuid4().hex}.json"
    dest = folder / name
    with dest.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _lex_normalize(text):
    """Exact lexical normalization from Script 27 (do not modify)."""
    text = text.lower()
    text = re.sub(r'https?://\S+', ' ', text)
    text = text.replace('ـ', '')
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    return ' '.join(text.split())


def _shingles(text):
    text = _lex_normalize(text)
    if not text:
        return {b'<EMPTY>'}
    if len(text) < SHINGLE_SIZE:
        return {text.encode('utf-8')}
    return {text[i:i+SHINGLE_SIZE].encode('utf-8')
            for i in range(len(text)-SHINGLE_SIZE+1)}


def _batch_signatures(work):
    """Runs in worker process; uses exactly MinHash.update's hash/permutation rules."""
    from datasketch import MinHash
    import numpy as np
    position, texts = work
    sig = np.empty((len(texts), NUM_PERM), dtype=np.uint64)
    for i, value in enumerate(texts):
        mh = MinHash(num_perm=NUM_PERM, seed=42)
        grams = _shingles(value)
        # update_batch is equivalent to calling update(g) on every gram,
        # but carries out the hash/permutation reductions using NumPy.
        if not hasattr(mh, 'update_batch'):
            raise RuntimeError('Your datasketch installation lacks MinHash.update_batch')
        # Prevent large allocations for extremely long Facebook posts.
        if len(grams) > 1000:
            grams = list(grams)
            for k in range(0, len(grams), 1000):
                mh.update_batch(grams[k:k+1000])
        else:
            mh.update_batch(list(grams))
        sig[i] = mh.hashvalues
    return position, sig


def verify_exact_signatures():
    """Stops immediately if vectorized hashing differs from original Script 27."""
    from datasketch import MinHash
    fixtures = ['', '#anxiety https://example.org/path', 'Hello WORLD! #abc',
                'مرحبا hello 123', 'a' * 3000, 'X\nY\nfollow us on facebook']
    for value in fixtures:
        serial = MinHash(num_perm=64, seed=42)
        for g in _shingles(value):
            serial.update(g)
        _, fast = _batch_signatures((0, [value]))
        require(bool((serial.hashvalues == fast[0]).all()),
                f'Exact MinHash verification FAILED for fixture {value[:50]!r}')
    print('[VERIFY] Batch MinHash signatures are identical to original serial update', flush=True)


def _signature_batches(texts, start):
    for left in range(start, len(texts), SIGNATURE_CHUNK):
        yield left, texts[left:left+SIGNATURE_CHUNK]


def build_signatures(texts, workers):
    import numpy as np
    n = len(texts)
    if has_progress(SIG_PROGRESS):
        progress = load_progress(SIG_PROGRESS)
        require(progress.get('n_texts') == n and progress.get('num_perm') == NUM_PERM,
                'Signature checkpoint incompatible with current input.')
        done = int(progress['completed'])
        require(SIGNATURES.exists(), 'Checkpoint exists but signatures.npy is missing')
        cache = np.lib.format.open_memmap(SIGNATURES, mode='r+')
        require(cache.shape == (n, NUM_PERM) and cache.dtype == np.uint64,
                'Signature cache shape/dtype incorrect')
    else:
        require(not SIGNATURES.exists(),
                'Signature file found without checkpoint: use --restart-minhash')
        cache = np.lib.format.open_memmap(
            SIGNATURES, mode='w+', dtype=np.uint64, shape=(n, NUM_PERM))
        done = 0
        save_progress(SIG_PROGRESS, {'n_texts': n, 'num_perm': NUM_PERM, 'completed': 0})
    require(0 <= done <= n, 'Invalid signature checkpoint')
    print(f'[SIGNATURES] {done:,}/{n:,} completed, workers={workers}', flush=True)
    if done == n:
        return cache
    start_time = time.perf_counter()

    def save_finished(position, arr):
        nonlocal done
        require(position == done, f'Noncontiguous signature block at {position}, expected {done}')
        end = position + len(arr)
        cache[position:end] = arr
        cache.flush()  # Every 512 texts: a crash cannot corrupt an acknowledged block.
        done = end
        save_progress(SIG_PROGRESS, {
            'n_texts': n, 'num_perm': NUM_PERM, 'completed': done})
        if done % 10000 < SIGNATURE_CHUNK or done == n:
            elapsed = (time.perf_counter() - start_time) / 60
            print(f'[SIGNATURES] {done:,}/{n:,} | session {elapsed:.1f} min', flush=True)

    tasks = _signature_batches(texts, done)
    if workers == 1:
        for task in tasks:
            save_finished(*_batch_signatures(task))
    else:
        # Bounded queue: do not pre-submit the entire corpus to workers.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            for _ in range(workers * 2):
                try:
                    pending.append(pool.submit(_batch_signatures, next(tasks)))
                except StopIteration:
                    break
            while pending:
                save_finished(*pending.popleft().result())
                try:
                    pending.append(pool.submit(_batch_signatures, next(tasks)))
                except StopIteration:
                    pass
    print('[SIGNATURES] Complete; cached for future restarts', flush=True)
    return cache


def _read_pair_count(file):
    import pyarrow.parquet as pq
    return int(pq.ParquetFile(file).metadata.num_rows)


def build_lsh_candidates(cache, n):
    """Same query-before-insert as source 27. Resume rebuilds index from signatures."""
    import pandas as pd
    from datasketch import MinHash, MinHashLSH
    
    if has_progress(LSH_PROGRESS):
        progress = load_progress(LSH_PROGRESS)
        require(progress.get('n_texts') == n and progress.get('chunk') == RETRIEVAL_CHUNK,
                'Incompatible LSH checkpoint.')
        completed = int(progress['completed'])
        pair_count = int(progress['candidate_pairs'])
        require(completed % RETRIEVAL_CHUNK == 0 or completed == n,
                'LSH checkpoint not at a shard boundary')
        for c in range(0, completed, RETRIEVAL_CHUNK):
            f = LEX / f'candidate_pairs_{c//RETRIEVAL_CHUNK:04d}.parquet'
            require(f.exists(), f'Completed checkpoint shard missing: {f}')
        require(sum(_read_pair_count(LEX / f'candidate_pairs_{c//RETRIEVAL_CHUNK:04d}.parquet')
                    for c in range(0, completed, RETRIEVAL_CHUNK)) == pair_count,
                'Completed shard row count does not match LSH checkpoint')
    else:
        # An old uncheckpointed runner may have saved shards. Never mix its
        # different shard boundaries with the new resumable shards.
        require(not list(LEX.glob('candidate_pairs_*.parquet')),
                'Found old nonresumable candidate shards. Stop the old process, '
                'then run --stage minhash --restart-minhash once.')
        completed = 0
        pair_count = 0
        save_progress(LSH_PROGRESS, {
            'n_texts': n, 'chunk': RETRIEVAL_CHUNK,
            'completed': 0, 'candidate_pairs': 0})

    print(f'[LSH] Rebuilding index for {completed:,} previously completed IDs from cached signatures', flush=True)
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    start_time = time.perf_counter()
    def hash_for(i):
        return MinHash(
        num_perm=NUM_PERM,
        seed=42,
        hashvalues=cache[i].copy(),
        scheme="affine32"
    )
    for i in range(completed):
        lsh.insert(str(i), hash_for(i))
        if (i+1) % 20000 == 0:
            print(f'[RESUME INDEX] {i+1:,}/{completed:,}', flush=True)

    for start in range(completed, n, RETRIEVAL_CHUNK):
        end = min(start + RETRIEVAL_CHUNK, n)
        pairs = []
        for i in range(start, end):
            mh = hash_for(i)
            for other in lsh.query(mh):
                j = int(other)
                if j != i:
                    pairs.append((j, i))
            lsh.insert(str(i), mh)
        dest = LEX / f'candidate_pairs_{start//RETRIEVAL_CHUNK:04d}.parquet'
        tmp = dest.with_name(dest.stem + '.tmp.parquet')
        pd.DataFrame(pairs, columns=['text_id_1', 'text_id_2']).to_parquet(tmp, index=False)
        os.replace(tmp, dest)
        pair_count += len(pairs)
        save_progress(LSH_PROGRESS, {
            'n_texts': n, 'chunk': RETRIEVAL_CHUNK,
            'completed': end, 'candidate_pairs': pair_count})
        if end % 10000 < RETRIEVAL_CHUNK or end == n:
            elapsed = (time.perf_counter() - start_time) / 60
            print(f'[LSH] {end:,}/{n:,} | candidate pairs={pair_count:,} | '
                  f'session {elapsed:.1f} min', flush=True)
    return pair_count, (n+RETRIEVAL_CHUNK-1)//RETRIEVAL_CHUNK


def clear_fast_minhash_outputs():
    require(not SEM.exists() or not list(SEM.iterdir()),
            'Refusing to restart: E5 results already present')
    allowed_names = {'text_id_mapping.parquet','minhash_summary.json',
                     'minhash_signatures.npy','minhash_signatures_progress.json',
                     'minhash_lsh_progress.json'}
    if LEX.exists():
        for p in LEX.iterdir():
            require(p.is_file() and (p.name in allowed_names or
                                    (p.name.startswith('candidate_pairs_') and
                                     p.name.endswith(('.parquet','.tmp.parquet')))),
                    f'Refusing to delete unknown file: {p}')
            p.unlink()
    for p in [SIGNATURES, SIG_PROGRESS, LSH_PROGRESS]:
        if p.exists(): p.unlink()
    for p in [SIG_PROGRESS, LSH_PROGRESS]:
        folder = _checkpoint_folder(p)
        if folder.exists():
            shutil.rmtree(folder)


def stage_minhash_fast(cleaned, manifest, restart, workers):
    import pandas as pd
    summary = LEX / 'minhash_summary.json'
    if summary.exists() and not restart:
        validate_minhash_mapping(cleaned)
        print('[REUSE] Full cleaned MinHash already complete', flush=True)
        print('[RESULT]', summary.read_text(encoding='utf-8'))
        return
    require(workers >= 1, 'workers must be >= 1')
    require(not restart or not (SEM.exists() and list(SEM.iterdir())),
            'E5 outputs exist; cannot restart MinHash')
    EXP.mkdir(parents=True, exist_ok=True)
    LEX.mkdir(parents=True, exist_ok=True)
    guard_manifest(manifest, create=True)
    if restart:
        clear_fast_minhash_outputs()
    verify_exact_signatures()
    texts = cleaned['normalized_text'].fillna('').astype(str).tolist()
    start = time.perf_counter()
    sig = build_signatures(texts, workers)
    count, shards = build_lsh_candidates(sig, len(texts))
    cleaned[['text_id','exact_family_id','normalized_text','post_count']].to_parquet(
        LEX/'text_id_mapping.parquet', index=False)
    validate_minhash_mapping(cleaned)
    report = {'unique_texts':len(texts), 'minhash_permutations':NUM_PERM,
              'lsh_threshold':LSH_THRESHOLD,'shingle_size':SHINGLE_SIZE,
              'lexical_candidate_pairs':count,'candidate_files':shards,
              'elapsed_minutes_this_session':round((time.perf_counter()-start)/60,2),
              'method':'faithful original Script 27; batch signatures and checkpointed LSH',
              'status':'complete'}
    save_json_atomic(summary, report)
    print('[RESULT]', json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check', 'minhash', 'e5', 'compare'], required=True)
    parser.add_argument('--restart-minhash', action='store_true',
                        help='Explicitly discard ONLY incomplete isolated MinHash shards and restart')
    parser.add_argument('--lexical-only', action='store_true',
                        help='Compare full MinHash candidate pools before cleaned E5 finishes')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='CPU workers for batched signatures (default: up to 4)')
    args = parser.parse_args()
    require(not args.restart_minhash or args.stage == 'minhash',
            '--restart-minhash applies only with --stage minhash')
    require(not args.lexical_only or args.stage == 'compare',
            '--lexical-only applies only with --stage compare')

    # Check that patch patterns match the ACTUAL uploaded original source code.
    make_derived_source('minhash')
    make_derived_source('e5')
    cleaned, n, altered = read_input_mappings()
    manifest = current_manifest(n, altered)
    if args.stage == 'check':
        guard_manifest(manifest)
        verify_exact_signatures()
        print('[CHECK] Source 27 and 32 exact path patches: PASS')
        print('[CHECK] Source text IDs and post counts: PASS')
        print('[CHECK] Existing combined-cleaned mapping reused; baseline remains untouched')
        print('[CHECK] Isolated full-corpus output:', EXP)
    elif args.stage == 'minhash':
        stage_minhash_fast(cleaned, manifest, args.restart_minhash, args.workers)
    elif args.stage == 'e5':
        stage_e5(cleaned, manifest)
    else:
        stage_compare(n, manifest, args.lexical_only)


if __name__ == '__main__':
    main()
