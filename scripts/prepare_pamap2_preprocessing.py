"""Reproduce the author notebook's PAMAP2 preprocessing and window split."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / 'data/raw/pamap2/PAMAP2_Dataset'
OUTPUT = ROOT / 'data/processed/pamap2-author-code'
CONFIG = json.loads((ROOT / 'configs/pamap2-data.json').read_text(encoding='utf-8'))
FEATURE_NAMES = CONFIG['channels']
FEATURE_IDXS = CONFIG['feature_column_indices']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    files = sorted(RAW_ROOT.rglob('*.dat'))
    if len(files) != 14:
        raise ValueError(f'Expected the author notebook inputs (14 .dat files), found {len(files)} under {RAW_ROOT}')
    frames = []
    for path in files:
        data = pd.read_csv(path, sep=r'\s+', header=None)
        if data.shape[1] != 54:
            raise ValueError(f'{path.name}: expected 54 columns, found {data.shape[1]}')
        selected = data.iloc[:, [1] + FEATURE_IDXS].copy()
        selected.columns = ['activity_id'] + FEATURE_NAMES
        frames.append(selected)
    data = pd.concat(frames, ignore_index=True)
    row_count_before = len(data)
    data = data[data.activity_id != 0].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    activity_ids = sorted(data.activity_id.unique())
    id2idx = {activity: index for index, activity in enumerate(activity_ids)}
    x = data[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = data.activity_id.map(id2idx).to_numpy(dtype=np.int32)
    win, step = CONFIG['window_length'], CONFIG['stride']
    starts = np.arange(0, len(x) - win + 1, step, dtype=np.int64)
    x_windows = np.stack([x[start:start + win] for start in starts]).astype(np.float32, copy=False)
    y_windows = np.asarray([np.bincount(y[start:start + win]).argmax() for start in starts], dtype=np.int32)
    window_ids = np.arange(len(starts), dtype=np.int64)
    train_ids, test_ids = train_test_split(
        window_ids, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=y_windows)
    x_train, x_test = x_windows[train_ids], x_windows[test_ids]
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape).astype(np.float32)
    x_test_s = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape).astype(np.float32)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts = {
        'X_train_s.npy': x_train_s, 'X_test_s.npy': x_test_s,
        'y_train.npy': y_windows[train_ids], 'y_test.npy': y_windows[test_ids],
        'train_window_ids.npy': train_ids, 'test_window_ids.npy': test_ids,
        'window_starts.npy': starts, 'window_labels.npy': y_windows,
    }
    artifact_hashes = {}
    for name, array in artifacts.items():
        target = OUTPUT / name
        np.save(target, array, allow_pickle=False)
        artifact_hashes[name] = sha256(target)
    (OUTPUT / 'labels.json').write_text(json.dumps({
        'activity_ids': [int(value) for value in activity_ids],
        'idx2id': {str(index): int(value) for index, value in enumerate(activity_ids)},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    artifact_hashes['labels.json'] = sha256(OUTPUT / 'labels.json')
    summary = {
        'dataset': CONFIG['dataset'], 'source_archive_sha256': CONFIG['archive_sha256'],
        'source_files': [{'path': path.relative_to(RAW_ROOT).as_posix(), 'bytes': path.stat().st_size,
                          'sha256': sha256(path)} for path in files],
        'source_rows_before_cleaning': row_count_before, 'clean_rows': len(data),
        'window_length': win, 'stride': step, 'window_count': len(starts),
        'train_windows': len(train_ids), 'test_windows': len(test_ids),
        'feature_count': len(FEATURE_NAMES), 'feature_names': FEATURE_NAMES,
        'activity_ids': [int(value) for value in activity_ids],
        'activity_windows': {str(index): int(np.sum(y_windows == index)) for index in range(len(activity_ids))},
        'split': 'sklearn train_test_split on windows, test_size=0.2, random_state=42, stratify=y_windows',
        'scaler_fit_scope': 'all timesteps in outer training windows',
        'scaler': {
            'n_samples_seen': int(scaler.n_samples_seen_),
            'mean': [float(value) for value in scaler.mean_],
            'variance': [float(value) for value in scaler.var_],
            'scale': [float(value) for value in scaler.scale_],
            'with_mean': bool(scaler.with_mean), 'with_std': bool(scaler.with_std),
        },
        'arrays_dtype': 'float32', 'artifact_sha256': artifact_hashes,
        'limitation': 'Protocol and Optional subject files are concatenated before windowing, matching author notebook. Windows can cross file boundaries; random window split is not subject-independent.',
    }
    (OUTPUT / 'preprocessing-summary.json').write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({key: summary[key] for key in ('source_rows_before_cleaning','clean_rows','window_count','train_windows','test_windows','feature_count','activity_ids')}, indent=2))


if __name__ == '__main__':
    main()
