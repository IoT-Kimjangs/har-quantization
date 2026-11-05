"""Evaluate the saved WISDM model on every prepared test window."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from inspect_wisdm_model import inspect
from prepare_wisdm_data import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    if args.batch_size < 1:
        raise ValueError('batch-size는 1 이상이어야 합니다.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError('출력 폴더가 비어 있지 않습니다. 다른 --output 경로를 지정해주세요.')
    model_metadata = inspect()
    config = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text())
    data_config = json.loads((ROOT / 'configs/wisdm-data.json').read_text())
    prepared = ROOT / config['output_directory']
    summary_path = prepared / 'preprocessing-summary.json'
    preparation = json.loads(summary_path.read_text())
    for name, expected in preparation['artifact_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'전처리 산출물 해시가 다릅니다: {name}')
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('KERAS_HOME', str(Path(sys.prefix) / '.keras'))
    os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix) / '.matplotlib'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import tensorflow as tf
    import keras
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from har_quantization.layers import AddPositionEmbedding
    tf.config.set_visible_devices([], 'GPU')
    labels = data_config['label_order']
    inputs = np.load(prepared / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prepared / 'y_test.npy', allow_pickle=False)
    window_ids = np.load(prepared / 'test_window_ids.npy', allow_pickle=False)
    if list(inputs.shape) != preparation['test_shape'] or inputs.shape[1:] != (80, 3):
        raise ValueError('평가 입력 형태가 준비 기록과 다릅니다.')
    if truth.shape != (len(inputs),) or window_ids.shape != truth.shape or len(inputs) == 0:
        raise ValueError('평가 입력과 정답·구간 번호 개수가 다릅니다.')
    if not np.isfinite(inputs).all() or not np.isin(truth, np.arange(len(labels))).all():
        raise ValueError('평가 입력 또는 정답에 잘못된 값이 있습니다.')
    model_config = json.loads((ROOT / 'configs/wisdm-model.json').read_text())
    model = keras.models.load_model(
        ROOT / model_config['local_path'],
        custom_objects={'AddPositionEmbedding': AddPositionEmbedding}, compile=False, safe_mode=True,
    )
    if model.input_shape != (None, 80, 3) or model.output_shape != (None, len(labels)):
        raise ValueError('모델 입출력 형태가 예상과 다릅니다.')
    probabilities = []
    for start in range(0, len(inputs), args.batch_size):
        scores = model(inputs[start:start + args.batch_size], training=False).numpy()
        if scores.shape != (min(args.batch_size, len(inputs) - start), len(labels)):
            raise ValueError('예측 출력 형태가 다릅니다.')
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('예측 확률에 잘못된 값이 있습니다.')
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-5)
        probabilities.append(scores)
    predicted = np.concatenate(probabilities).argmax(axis=1)
    class_ids = np.arange(len(labels))
    matrix = confusion_matrix(truth, predicted, labels=class_ids)
    accuracy = float(accuracy_score(truth, predicted))
    macro_f1 = float(f1_score(truth, predicted, labels=class_ids, average='macro', zero_division=0))
    # Verify the metric calculation independently from the confusion matrix.
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    manual_f1 = np.divide(2 * matrix.diagonal(), denominator,
                          out=np.zeros(len(labels), dtype=float), where=denominator != 0).mean()
    np.testing.assert_allclose(accuracy, np.trace(matrix) / matrix.sum(), atol=1e-12)
    np.testing.assert_allclose(macro_f1, manual_f1, atol=1e-12)
    paper = {'accuracy_percent': 96.771, 'macro_f1_percent': 95.458,
             'doi': '10.1038/s41598-026-62744-4', 'pdf_page': 11, 'table': 2}
    report = {
        'dataset': 'WISDM Activity Prediction v1.1', 'platform': platform.platform(),
        'architecture': platform.machine(), 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('tensorflow', 'keras', 'numpy', 'scikit-learn')},
        'model_sha256': model_metadata['sha256'], 'raw_sha256': preparation['raw_sha256'],
        'preprocessing_summary_sha256': digest(summary_path),
        'evaluation_artifact_sha256': {name: preparation['artifact_sha256'][name] for name in ('X_test_s.npy', 'y_test.npy', 'test_window_ids.npy', 'scaler.json')},
        'method': 'model(x, training=False)', 'batch_size': args.batch_size, 'device': 'CPU',
        'label_order': labels, 'evaluated_windows': len(truth),
        'correct_windows': int(np.trace(matrix)), 'incorrect_windows': int(matrix.sum() - np.trace(matrix)),
        'accuracy': accuracy, 'macro_f1': macro_f1,
        'accuracy_percent': 100 * accuracy, 'macro_f1_percent': 100 * macro_f1,
        'paper_reference': paper,
        'difference_percentage_points': {'accuracy': 100 * accuracy - paper['accuracy_percent'],
                                         'macro_f1': 100 * macro_f1 - paper['macro_f1_percent']},
        'classification_report': classification_report(truth, predicted, labels=class_ids, target_names=labels, output_dict=True, zero_division=0),
        'confusion_matrix': matrix.tolist(),
        'matches_saved_model_training_statistics': preparation['matches_saved_model_training_statistics'],
        'limitations': ['Scaler and split recalculated from public code; original training artifacts unavailable.',
                        'Window-wise split: subjects and overlapping source samples appear in both partitions.'],
        'metric_cross_check': 'passed', 'training_executed': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['test_index', 'window_id', 'true_label', 'predicted_label'])
        writer.writerows((i, int(window_ids[i]), int(truth[i]), int(predicted[i])) for i in range(len(truth)))
    report['predictions_sha256'] = digest(output / 'predictions.csv')
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('evaluated_windows', 'correct_windows', 'incorrect_windows',
        'accuracy_percent', 'macro_f1_percent', 'difference_percentage_points')}, indent=2))
    print(f'결과: {output}')


if __name__ == '__main__':
    main()
