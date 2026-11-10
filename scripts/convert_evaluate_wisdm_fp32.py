"""Convert WISDM to TFLite FP32 and compare every test output against Keras."""
import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from inspect_wisdm_model import inspect
from prepare_wisdm_data import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    target = ROOT / 'models/wisdm_transformer_fp32.tflite'
    if target.exists() or (output.exists() and any(output.iterdir())):
        raise ValueError('기존 모델 또는 결과가 있습니다. 기존 파일을 먼저 확인해주세요.')
    model_metadata = inspect()
    manifest = json.loads((ROOT / 'configs/wisdm-model.json').read_text())
    config = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text())
    labels = json.loads((ROOT / 'configs/wisdm-data.json').read_text())['label_order']
    prepared = ROOT / config['output_directory']
    preparation = json.loads((prepared / 'preprocessing-summary.json').read_text())
    for name, expected in preparation['artifact_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'전처리 파일 해시가 다릅니다: {name}')
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('KERAS_HOME', str(Path(sys.prefix) / '.keras'))
    os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix) / '.matplotlib'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    sys.path.insert(0, str(ROOT / 'src'))
    import numpy as np
    import tensorflow as tf
    import keras
    from ai_edge_litert.interpreter import Interpreter
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from har_quantization.layers import AddPositionEmbedding
    tf.config.set_visible_devices([], 'GPU')
    inputs = np.load(prepared / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prepared / 'y_test.npy', allow_pickle=False)
    window_ids = np.load(prepared / 'test_window_ids.npy', allow_pickle=False)
    if list(inputs.shape) != preparation['test_shape'] or inputs.shape[1:] != (80, 3):
        raise ValueError('평가 입력 형태가 다릅니다.')
    if truth.shape != (len(inputs),) or window_ids.shape != truth.shape or not len(truth):
        raise ValueError('평가 입력·정답·구간 수가 다릅니다.')
    if not np.isfinite(inputs).all() or not np.isin(truth, np.arange(len(labels))).all():
        raise ValueError('잘못된 평가 입력 또는 라벨이 있습니다.')
    model = keras.models.load_model(ROOT / manifest['local_path'],
        custom_objects={'AddPositionEmbedding': AddPositionEmbedding}, compile=False, safe_mode=True)
    with tempfile.TemporaryDirectory(prefix='wisdm-export-', dir=ROOT / 'models') as folder:
        export = Path(folder) / 'saved_model'
        model.export(str(export), format='tf_saved_model', verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(export))
        converter.optimizations = []
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converted = converter.convert()
    # No SELECT_TF_OPS fallback and no quantization are enabled.
    target.write_bytes(converted)
    interpreter = Interpreter(model_content=converted, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if inp['dtype'] != np.float32 or out['dtype'] != np.float32:
        raise ValueError('입출력이 float32가 아닙니다.')
    if list(inp['shape']) != [1, 80, 3] or list(out['shape']) != [1, len(labels)]:
        raise ValueError('변환된 모델 입출력 형태가 다릅니다.')
    tensor_types = Counter(np.dtype(t['dtype']).name for t in interpreter.get_tensor_details())
    if 'float16' in tensor_types or 'int8' in tensor_types:
        raise ValueError('FP32 모델에서 예상하지 않은 FP16/INT8 텐서를 찾았습니다.')
    keras_scores = np.concatenate([model(inputs[i:i + 256], training=False).numpy()
                                   for i in range(0, len(inputs), 256)])
    lite_scores = np.empty((len(inputs), len(labels)), dtype=np.float32)
    for i in range(len(inputs)):
        interpreter.set_tensor(inp['index'], inputs[i:i + 1])
        interpreter.invoke()
        lite_scores[i] = interpreter.get_tensor(out['index'])[0]
    for scores in (keras_scores, lite_scores):
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('잘못된 출력 확률이 있습니다.')
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-5)
    reference_labels = keras_scores.argmax(axis=1)
    lite_labels = lite_scores.argmax(axis=1)
    error = np.abs(keras_scores.astype(np.float64) - lite_scores.astype(np.float64))
    class_ids = np.arange(len(labels))

    def metrics(predicted):
        matrix = confusion_matrix(truth, predicted, labels=class_ids)
        accuracy = float(accuracy_score(truth, predicted))
        f1 = float(f1_score(truth, predicted, labels=class_ids, average='macro', zero_division=0))
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        manual = np.divide(2 * matrix.diagonal(), denominator,
                           out=np.zeros(len(labels)), where=denominator != 0).mean()
        np.testing.assert_allclose(f1, manual, atol=1e-12)
        np.testing.assert_allclose(accuracy, np.trace(matrix) / len(truth), atol=1e-12)
        return {'accuracy_percent': accuracy * 100, 'macro_f1_percent': f1 * 100,
                'correct_windows': int(np.trace(matrix)), 'confusion_matrix': matrix.tolist()}

    report = {
        'platform': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('tensorflow', 'keras', 'ai-edge-litert', 'numpy', 'scikit-learn')},
        'keras_sha256': model_metadata['sha256'], 'tflite_sha256': digest(target),
        'keras_bytes': model_metadata['size_bytes'], 'tflite_bytes': len(converted),
        'keras_kib': model_metadata['size_bytes'] / 1024, 'tflite_kib': len(converted) / 1024,
        'evaluated_windows': len(truth), 'label_order': labels,
        'input_shape': inp['shape'].tolist(), 'output_shape': out['shape'].tolist(),
        'input_dtype': str(np.dtype(inp['dtype'])), 'output_dtype': str(np.dtype(out['dtype'])),
        'tensor_dtypes': dict(tensor_types), 'optimizations': [], 'select_tf_ops_enabled': False,
        'interpreter_threads': 1, 'device': 'CPU',
        'keras_reference': metrics(reference_labels), 'tflite_fp32': metrics(lite_labels),
        'prediction_agreement_percent': float(np.mean(reference_labels == lite_labels) * 100),
        'disagreement_windows': int(np.count_nonzero(reference_labels != lite_labels)),
        'max_absolute_probability_error': float(error.max()), 'mean_absolute_probability_error': float(error.mean()),
        'evaluation_artifact_sha256': {name: preparation['artifact_sha256'][name] for name in ('X_test_s.npy', 'y_test.npy', 'test_window_ids.npy', 'scaler.json')},
        'matches_saved_model_training_statistics': preparation['matches_saved_model_training_statistics'],
        'training_executed': False, 'latency_measured': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['test_index', 'window_id', 'true_label', 'keras_label', 'tflite_label', 'max_absolute_probability_error'])
        writer.writerows((i, int(window_ids[i]), int(truth[i]), int(reference_labels[i]),
                          int(lite_labels[i]), float(error[i].max())) for i in range(len(inputs)))
    report['predictions_sha256'] = digest(output / 'predictions.csv')
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
