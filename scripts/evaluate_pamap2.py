"""Evaluate the stored author PAMAP2 model on the reproduced test split."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    config = json.loads((ROOT / 'configs/pamap2-model.json').read_text(encoding='utf-8'))
    model_path = ROOT / config['local_path']
    if sha256(model_path) != config['sha256']:
        raise ValueError('Stored PAMAP2 model hash differs from its pinned config')
    output = ROOT / 'results/2026-11-10-windows-pamap2-fp32'
    output.mkdir(parents=True, exist_ok=True)
    prep_dir = ROOT / 'data/processed/pamap2-author-code'
    prep_summary_path = prep_dir / 'preprocessing-summary.json'
    prep = json.loads(prep_summary_path.read_text(encoding='utf-8'))
    public_summary_path = output / 'preprocessing-summary.json'
    shutil.copyfile(prep_summary_path, public_summary_path)
    for name, expected in prep['artifact_sha256'].items():
        if sha256(prep_dir / name) != expected:
            raise ValueError(f'Preprocessing artifact hash mismatch: {name}')
    x = np.load(prep_dir / 'X_test_s.npy', allow_pickle=False)
    y_true = np.load(prep_dir / 'y_test.npy', allow_pickle=False)
    window_ids = np.load(prep_dir / 'test_window_ids.npy', allow_pickle=False)
    model = keras.models.load_model(model_path, custom_objects={'AddPositionEmbedding': __import__(
        'har_quantization.layers', fromlist=['AddPositionEmbedding']).AddPositionEmbedding},
        compile=False, safe_mode=True)
    if list(model.input_shape[1:]) != [200, 19] or model.output_shape[-1] != len(prep['activity_ids']):
        raise ValueError(f'Unexpected model signature {model.input_shape} -> {model.output_shape}')
    probs = np.asarray(model(x, training=False).numpy())
    pred = probs.argmax(axis=1).astype(np.int32)
    labels = [f'activity_{value}' for value in prep['activity_ids']]
    accuracy = float(accuracy_score(y_true, pred))
    macro_f1 = float(f1_score(y_true, pred, average='macro', zero_division=0))
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(labels)))
    predictions = pd.DataFrame({'test_index': np.arange(len(y_true)), 'window_id': window_ids,
                                'y_true': y_true, 'keras_label': pred})
    predictions.to_csv(output / 'predictions.csv', index=False, lineterminator='\n')
    metrics = {
        'dataset': 'PAMAP2 Physical Activity Monitoring', 'platform': platform.platform(),
        'architecture': platform.machine(), 'python': platform.python_version(),
        'packages': {'tensorflow': tf.__version__, 'keras': keras.__version__,
                     'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'pandas': pd.__version__},
        'model_parameter_count': int(model.count_params()),
        'model_sha256': config['sha256'], 'raw_archive_sha256': prep['source_archive_sha256'],
        'preprocessing_summary_path': 'preprocessing-summary.json',
        'preprocessing_summary_sha256': sha256(public_summary_path),
        'evaluated_windows': len(y_true), 'correct_windows': int((y_true == pred).sum()),
        'accuracy': accuracy, 'macro_f1': macro_f1,
        'accuracy_percent': accuracy * 100, 'macro_f1_percent': macro_f1 * 100,
        'paper_reference': {'accuracy_percent': 98.480, 'macro_f1_percent': None,
                            'note': 'Paper benchmark uses its own experiment split; comparison is contextual only.'},
        'difference_percentage_points': {'accuracy': accuracy * 100 - 98.480},
        'classification_report': classification_report(y_true, pred, labels=np.arange(len(labels)),
                                                        target_names=labels, output_dict=True, zero_division=0),
        'confusion_matrix': cm.tolist(), 'labels': labels,
        'matches_saved_model_training_statistics': 'unverified',
        'limitations': [
            'Scaler and split training artifacts are unavailable; scaler was refit on this run training windows.',
            'Author preprocessing concatenates Optional and Protocol files before windowing; windows can cross subject/file boundaries.',
            'Random window-level split is not subject-independent and overlapping samples can cross partitions.',
            'No training was performed; accuracy is for a stored model evaluated on this reproduced split.',
        ],
        'training_executed': False, 'metric_cross_check': 'passed',
        'prediction_sha256': sha256(output / 'predictions.csv'),
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: metrics[key] for key in ('evaluated_windows','correct_windows','accuracy_percent','macro_f1_percent')}, indent=2))


if __name__ == '__main__':
    main()
