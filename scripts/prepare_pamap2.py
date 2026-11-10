"""Prepare official PAMAP2 raw files and reproduce the author's window split/scaler."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_checked(archive, destination):
    """Extract data/docs/nested ZIP files only; verify CRC and reject path traversal."""
    nested = []
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts or '\\' in item.filename:
                raise ValueError('허용하지 않는 ZIP 경로입니다.')
            if item.is_dir() or relative.suffix.lower() not in ('.dat', '.pdf', '.txt', '.zip'):
                continue
            target = destination / relative
            if target.exists():
                crc = 0
                with target.open('rb') as stream:
                    while block := stream.read(1024 * 1024):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != item.file_size or crc != item.CRC:
                    raise ValueError(f'기존 원본이 ZIP 내용과 다릅니다: {relative}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.partial')
                with z.open(item) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                partial.replace(target)
            if target.suffix.lower() == '.zip':
                nested.append(target)
    for path in nested:
        extract_checked(path, path.parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Official UCI outer ZIP or PAMAP2_Dataset.zip')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/pamap2.json').read_text())
    source = args.archive.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'다운로드한 데이터 ZIP이 없습니다: {source}')
    if sha256(ROOT / config['model_path']) != config['model_sha256']:
        raise ValueError('저자 모델의 SHA-256이 다릅니다.')
    raw_root = ROOT / config['raw_directory']
    processed = ROOT / config['processed_directory']
    if processed.exists() and any(processed.iterdir()):
        raise ValueError('기존 전처리 결과가 있습니다. 덮어쓰지 않습니다.')
    raw_root.mkdir(parents=True, exist_ok=True)
    stored_archive = raw_root / source.name
    archive_hash = sha256(source)
    if config.get('accepted_archive_sha256') and archive_hash not in config['accepted_archive_sha256'].values():
        raise ValueError('기록된 공식 ZIP과 해시가 다릅니다. 배포 파일 변경 여부를 먼저 확인해주세요.')
    if source != stored_archive.resolve():
        if stored_archive.exists():
            if sha256(stored_archive) != archive_hash:
                raise ValueError('보관한 ZIP과 선택한 ZIP이 다릅니다.')
        else:
            shutil.copyfile(source, stored_archive)
    extract_checked(stored_archive, raw_root)
    files = sorted(raw_root.rglob('*.dat'), key=lambda p: p.relative_to(raw_root).as_posix())
    if len(files) != config['expected_dat_files']:
        raise ValueError(f'원본 .dat는 Optional+Protocol 14개여야 합니다. 현재 {len(files)}개입니다.')
    if {p.parent.name for p in files} != {'Optional', 'Protocol'}:
        raise ValueError('Optional과 Protocol 두 종류의 데이터가 필요합니다.')
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    processed.mkdir(parents=True)
    clean_path = processed / 'clean-selected-columns.bin'
    file_records = []
    selected_columns = [1] + config['feature_columns_zero_based']
    rows_so_far = 0
    activity_set = set()
    with clean_path.open('wb') as clean:
        for path in files:
            raw_count = zero_count = invalid_count = kept_count = 0
            for chunk in pd.read_csv(path, sep=r'\s+', header=None, usecols=selected_columns,
                                     dtype=np.float64, chunksize=100000):
                values = chunk.to_numpy()
                raw_count += len(values)
                nonzero = values[:, 0] != 0
                zero_count += int(np.count_nonzero(~nonzero))
                values = values[nonzero]
                finite = np.isfinite(values).all(axis=1)
                invalid_count += int(np.count_nonzero(~finite))
                values = np.ascontiguousarray(values[finite], dtype=np.float64)
                kept_count += len(values)
                activity_set.update(int(v) for v in np.unique(values[:, 0]))
                values.tofile(clean)
            file_records.append({'path': path.relative_to(raw_root).as_posix(), 'sha256': sha256(path),
                                 'raw_rows': raw_count, 'activity_zero_rows': zero_count,
                                 'invalid_selected_rows': invalid_count, 'kept_rows': kept_count,
                                 'clean_start': rows_so_far, 'clean_end': rows_so_far + kept_count})
            rows_so_far += kept_count
            print(f'{path.parent.name}/{path.name}: {kept_count}개 유효 행', flush=True)
    activities = sorted(activity_set)
    if activities != config['activity_ids']:
        raise ValueError(f'모델과 맞는 18개 행동이 아닙니다: {activities}')
    clean = np.memmap(clean_path, dtype=np.float64, mode='r', shape=(rows_so_far, len(selected_columns)))
    length, step = config['window_length'], config['window_step']
    starts = np.arange(0, rows_so_far - length + 1, step, dtype=np.int64)
    # Preserve concatenation across file/subject/activity boundaries, as in the public notebook.
    X = np.lib.format.open_memmap(processed / 'X_windows.npy', mode='w+', dtype=np.float32,
                                 shape=(len(starts), length, len(config['feature_names'])))
    y = np.empty(len(starts), dtype=np.int32)
    mapping = {activity: i for i, activity in enumerate(activities)}
    encoded = np.array([mapping[int(value)] for value in clean[:, 0]], dtype=np.int32)
    for i, start in enumerate(starts):
        X[i] = clean[start:start + length, 1:]
        y[i] = np.bincount(encoded[start:start + length]).argmax()
    X.flush()
    ids = np.arange(len(X), dtype=np.int64)
    train_ids, test_ids = train_test_split(ids, test_size=config['test_size'],
        random_state=config['seed'], stratify=y)
    train = np.ascontiguousarray(X[train_ids])
    test = np.ascontiguousarray(X[test_ids])
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    test_s = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    del train, test
    arrays = {'X_train_s.npy': train_s, 'X_test_s.npy': test_s, 'y_train.npy': y[train_ids],
              'y_test.npy': y[test_ids], 'train_window_ids.npy': train_ids, 'test_window_ids.npy': test_ids,
              'window_starts.npy': starts}
    for name, array in arrays.items():
        np.save(processed / name, array, allow_pickle=False)
    stats = {'feature_order': config['feature_names'], 'mean': scaler.mean_.tolist(),
             'scale': scaler.scale_.tolist(), 'variance': scaler.var_.tolist(),
             'n_samples_seen': int(scaler.n_samples_seen_),
             'provenance': 'recalculated_from_public_code', 'matches_saved_model_training_statistics': 'unverified'}
    (processed / 'scaler.json').write_text(json.dumps(stats, indent=2) + '\n')
    if np.intersect1d(train_ids, test_ids).size or not np.array_equal(np.sort(np.r_[train_ids, test_ids]), ids):
        raise ValueError('분할 인덱스에 중복·누락이 있습니다.')
    if not np.isfinite(train_s).all() or not np.isfinite(test_s).all():
        raise ValueError('표준화 결과에 비유한 값이 있습니다.')
    # Verify a held-out batch against the scaler equation, independent of sklearn transform.
    raw_test = np.asarray(X[test_ids[:128]], dtype=np.float64)
    np.testing.assert_allclose(test_s[:128], ((raw_test - scaler.mean_) / scaler.scale_).astype(np.float32), rtol=2e-6, atol=2e-6)
    in_test = np.zeros(len(X), dtype=bool); in_test[test_ids] = True
    boundaries = [r['clean_end'] for r in file_records[:-1]]
    crossing = sum(bool(any(start < end < start + length for end in boundaries)) for start in starts)
    report = {'dataset': 'PAMAP2', 'dataset_url': config['dataset_url'], 'archive_sha256': archive_hash,
        'source_commit': config['source_commit'], 'model_sha256': config['model_sha256'],
        'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'raw_files': file_records, 'clean_rows': rows_so_far, 'window_shape': list(X.shape),
        'train_shape': list(train_s.shape), 'test_shape': list(test_s.shape), 'activity_ids': activities,
        'class_counts': {str(a): int(np.count_nonzero(y == i)) for i,a in enumerate(activities)},
        'windows_crossing_file_boundaries': crossing,
        'adjacent_train_test_overlapping_pairs': int(np.count_nonzero(in_test[:-1] != in_test[1:])),
        'subject_boundaries_preserved': False, 'scaler': stats,
        'matches_saved_model_training_statistics': 'unverified',
        'artifact_sha256': {name: sha256(processed / name) for name in [*arrays, 'scaler.json']},
        'verification': {'disjoint_window_ids': 'passed', 'finite_arrays': 'passed', 'manual_test_transform': 'passed'}}
    (processed / 'preprocessing-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    result = ROOT / 'results/pamap2-preparation';result.mkdir(parents=True, exist_ok=True)
    (result / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ('clean_rows','window_shape','train_shape','test_shape','windows_crossing_file_boundaries')},indent=2))


if __name__ == '__main__':
    main()
