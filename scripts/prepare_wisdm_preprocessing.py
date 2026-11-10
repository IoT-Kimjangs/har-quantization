"""Recalculate WISDM splits and scaler from the published baseline procedure."""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare_wisdm_data import digest

ROOT = Path(__file__).resolve().parents[1]


def load_raw(path, labels):
    rows = []
    with path.open() as stream:
        for line in stream:
            value = line.strip()
            if not value:
                continue
            parts = value.removesuffix(';').split(',')
            if len(parts) == 6:
                rows.append(parts)
    frame = pd.DataFrame(rows, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    for column in ('timestamp', 'x', 'y', 'z'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame.dropna(inplace=True)
    frame = frame[(frame.timestamp > 0) & frame.activity.isin(labels)]
    frame.reset_index(drop=True, inplace=True)
    # Preserve the author's string user IDs and default pandas sorting behavior.
    frame.sort_values(['user', 'timestamp'], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    if not np.isfinite(frame[['timestamp', 'x', 'y', 'z']].to_numpy()).all():
        raise ValueError('비유한 숫자가 있습니다. 원본 절차 적용 전에 확인하세요.')
    return frame


def make_windows(frame, config, labels):
    inputs, targets, users, starts = [], [], [], []
    mixed_count = 0
    label2idx = {label: index for index, label in enumerate(labels)}
    for user, group in frame.groupby('user', sort=False):
        values = group[config['feature_order']].values.astype(np.float32)
        activities = group.activity.map(label2idx).values.astype(np.int32)
        for start in range(0, len(values) - config['window_length'] + 1, config['window_step']):
            end = start + config['window_length']
            window_labels = activities[start:end]
            inputs.append(values[start:end])
            targets.append(np.bincount(window_labels).argmax())
            users.append(user)
            starts.append(start)
            mixed_count += int(np.unique(window_labels).size > 1)
    return (np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.int32),
            np.array(users), np.array(starts, dtype=np.int64), mixed_count)


def label_counts(values, labels):
    counts = np.bincount(values, minlength=len(labels))
    return dict(zip(labels, counts.tolist()))


def save_checked(path, writer):
    temporary = path.with_name(path.name + '.partial')
    try:
        with temporary.open('wb') as stream:
            writer(stream)
        if path.exists() and digest(path) != digest(temporary):
            raise ValueError(f'기존 결과와 다릅니다. 덮어쓰지 않습니다: {path}')
        if path.exists():
            temporary.unlink()
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    config = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text())
    data = json.loads((ROOT / 'configs/wisdm-data.json').read_text())
    raw = ROOT / data['local_directory'] / data['raw_file']
    if digest(raw) != data['files'][data['raw_file']]:
        raise ValueError('원본 데이터 해시가 다릅니다.')
    labels = data['label_order']
    if config['feature_order'] != data['feature_order']:
        raise ValueError('센서 채널 순서가 다릅니다.')
    frame = load_raw(raw, labels)
    windows, targets, users, starts, mixed_count = make_windows(frame, config, labels)
    ids = np.arange(len(windows), dtype=np.int64)
    train_ids, test_ids = train_test_split(
        ids, test_size=config['test_size'], random_state=config['random_state'], stratify=targets,
    )
    scaler = StandardScaler()
    train = windows[train_ids]
    test = windows[test_ids]
    train_scaled = scaler.fit_transform(train.reshape(-1, 3)).reshape(train.shape)
    test_scaled = scaler.transform(test.reshape(-1, 3)).reshape(test.shape)

    # Independent numerical checks: train-only moments, test transform, split coverage.
    flat = train.reshape(-1, 3).astype(np.float64)
    np.testing.assert_allclose(scaler.mean_, flat.mean(axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scaler.var_, flat.var(axis=0), rtol=1e-10, atol=1e-10)
    manual_test = ((test.astype(np.float64) - scaler.mean_) / scaler.scale_).astype(np.float32)
    np.testing.assert_allclose(test_scaled, manual_test, rtol=2e-6, atol=2e-6)
    if len(np.intersect1d(train_ids, test_ids)) or not np.array_equal(np.sort(np.r_[train_ids, test_ids]), ids):
        raise ValueError('분할 인덱스가 겹치거나 누락됐습니다.')
    if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
        raise ValueError('표준화 결과에 비유한 값이 있습니다.')

    # Distinct windows may still share raw sensor samples after random splitting.
    in_test = np.zeros(len(ids), dtype=bool)
    in_test[test_ids] = True
    adjacent = (users[:-1] == users[1:]) & (starts[1:] - starts[:-1] < config['window_length'])
    crossing_pairs = int(np.count_nonzero(adjacent & (in_test[:-1] != in_test[1:])))
    arrays = {
        'X_train_s.npy': train_scaled, 'X_test_s.npy': test_scaled,
        'y_train.npy': targets[train_ids], 'y_test.npy': targets[test_ids],
        'train_window_ids.npy': train_ids, 'test_window_ids.npy': test_ids,
        'window_users.npy': users, 'window_starts.npy': starts,
    }
    output = ROOT / config['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        save_checked(output / name, lambda stream, array=array: np.save(stream, array, allow_pickle=False))
    stats = {
        'feature_order': config['feature_order'], 'mean': scaler.mean_.tolist(),
        'variance': scaler.var_.tolist(), 'scale': scaler.scale_.tolist(),
        'n_samples_seen': int(scaler.n_samples_seen_), 'with_mean': True, 'with_std': True,
        'provenance': 'recalculated_from_public_code',
        'matches_saved_model_training_statistics': 'unverified',
    }
    save_checked(output / 'scaler.json', lambda stream: stream.write((json.dumps(stats, indent=2) + '\n').encode()))
    report = {
        'provenance': 'recalculated_from_public_code',
        'matches_saved_model_training_statistics': 'unverified',
        'source_commit': config['source_commit'], 'notebook_sha256': config['notebook_sha256'],
        'raw_sha256': digest(raw),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
        'clean_rows': len(frame), 'window_shape': list(windows.shape),
        'train_shape': list(train_scaled.shape), 'test_shape': list(test_scaled.shape),
        'array_dtype': str(train_scaled.dtype),
        'train_label_counts': label_counts(targets[train_ids], labels),
        'test_label_counts': label_counts(targets[test_ids], labels),
        'mixed_activity_window_count': mixed_count,
        'shared_train_test_user_count': len(set(users[train_ids]) & set(users[test_ids])),
        'overlapping_train_test_adjacent_window_pairs': crossing_pairs,
        'scaler': stats,
        'artifact_sha256': {name: digest(output / name) for name in [*arrays, 'scaler.json']},
        'verification': {'train_only_moments': 'passed', 'manual_test_transform': 'passed',
                         'disjoint_window_ids_and_full_coverage': 'passed', 'finite_outputs': 'passed'},
        'model_loaded': False, 'prediction_executed': False,
    }
    report_path = output / 'preprocessing-summary.json'
    save_checked(report_path, lambda stream: stream.write((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'전처리 실패: {error}', file=sys.stderr)
        raise SystemExit(1)
