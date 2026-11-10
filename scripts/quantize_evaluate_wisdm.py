"""Create three WISDM PTQ variants and evaluate the complete prepared test set."""
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
VARIANTS = ('fp16', 'int8_dynamic', 'int8_full')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp32-result', type=Path,
                        default=Path('results/2026-11-09-macos-wisdm-tflite-fp32'))
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    baseline_dir = args.fp32_result if args.fp32_result.is_absolute() else ROOT / args.fp32_result
    paths = {name: ROOT / f'models/wisdm_transformer_{name}.tflite' for name in VARIANTS}
    if (output.exists() and any(output.iterdir())) or any(p.exists() for p in paths.values()):
        raise ValueError('기존 양자화 모델 또는 결과가 있습니다. 기존 파일을 먼저 확인해주세요.')
    original = inspect()
    config = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text())
    model_config = json.loads((ROOT / 'configs/wisdm-model.json').read_text())
    labels = json.loads((ROOT / 'configs/wisdm-data.json').read_text())['label_order']
    prepared = ROOT / config['output_directory']
    preparation = json.loads((prepared / 'preprocessing-summary.json').read_text())
    for name, expected in preparation['artifact_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'전처리 산출물 해시가 다릅니다: {name}')
    baseline = json.loads((baseline_dir / 'metrics.json').read_text())
    if baseline['keras_sha256'] != original['sha256']:
        raise ValueError('FP32 평가의 원본 모델이 다릅니다.')
    for name, expected in baseline['evaluation_artifact_sha256'].items():
        if preparation['artifact_sha256'][name] != expected:
            raise ValueError(f'FP32 평가와 전처리 조건이 다릅니다: {name}')
    if digest(ROOT / 'models/wisdm_transformer_fp32.tflite') != baseline['tflite_sha256']:
        raise ValueError('FP32 모델 파일 해시가 다릅니다.')
    if digest(baseline_dir / 'predictions.csv') != baseline['predictions_sha256']:
        raise ValueError('FP32 예측 기록 해시가 다릅니다.')
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
    train = np.load(prepared / 'X_train_s.npy', allow_pickle=False)
    inputs = np.load(prepared / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prepared / 'y_test.npy', allow_pickle=False)
    train_ids = np.load(prepared / 'train_window_ids.npy', allow_pickle=False)
    test_ids = np.load(prepared / 'test_window_ids.npy', allow_pickle=False)
    if list(train.shape) != preparation['train_shape'] or list(inputs.shape) != preparation['test_shape']:
        raise ValueError('준비 기록과 입력 형태가 다릅니다.')
    if truth.shape != (len(inputs),) or test_ids.shape != truth.shape or train_ids.shape != (len(train),):
        raise ValueError('입력·정답·구간 번호 개수가 다릅니다.')
    if not np.isfinite(train).all() or not np.isfinite(inputs).all():
        raise ValueError('입력에 비유한 값이 있습니다.')
    with (baseline_dir / 'predictions.csv').open() as stream:
        prior = list(csv.DictReader(stream))
    if len(prior) != len(truth) or not all(int(r['window_id']) == int(test_ids[i]) and
            int(r['true_label']) == int(truth[i]) for i, r in enumerate(prior)):
        raise ValueError('FP32 평가의 입력·정답 순서가 다릅니다.')
    baseline_prediction = np.array([int(r['tflite_label']) for r in prior])
    selected = np.random.RandomState(42).choice(len(train), min(300, len(train)), replace=False)
    if np.intersect1d(train_ids[selected], test_ids).size:
        raise ValueError('INT8 보정 구간 번호에 평가 구간이 포함돼 있습니다.')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'calibration.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['train_index', 'window_id'])
        writer.writerows((int(i), int(train_ids[i])) for i in selected)

    def representative_data():
        for i in selected:
            yield [train[i:i + 1].astype(np.float32)]

    def evaluate(path, name):
        interpreter = Interpreter(model_path=str(path), num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        expected_dtype = np.int8 if name == 'int8_full' else np.float32
        if inp['dtype'] != expected_dtype or out['dtype'] != expected_dtype:
            raise ValueError('예상 입출력 dtype과 다릅니다.')
        if list(inp['shape']) != [1, 80, 3] or list(out['shape']) != [1, len(labels)]:
            raise ValueError('변환된 모델의 입출력 형태가 다릅니다.')
        dtype_counts = Counter(np.dtype(t['dtype']).name for t in interpreter.get_tensor_details())
        if name == 'fp16' and not dtype_counts['float16']:
            raise ValueError('FP16 텐서가 없습니다.')
        if name.startswith('int8') and not dtype_counts['int8']:
            raise ValueError('INT8 텐서가 없습니다.')
        scores = np.empty((len(inputs), len(labels)), dtype=np.float32)
        clipped = 0
        input_scale, input_zero = inp['quantization']
        output_scale, output_zero = out['quantization']
        if name == 'int8_full' and (input_scale <= 0 or output_scale <= 0):
            raise ValueError('INT8 입출력 quantization scale이 유효하지 않습니다.')
        for i in range(len(inputs)):
            sample = inputs[i:i + 1]
            if name == 'int8_full':
                quantized = np.rint(sample / input_scale + input_zero)
                clipped += int(np.count_nonzero((quantized < -128) | (quantized > 127)))
                sample = np.clip(quantized, -128, 127).astype(np.int8)
            interpreter.set_tensor(inp['index'], sample)
            interpreter.invoke()
            result = interpreter.get_tensor(out['index'])[0]
            if name == 'int8_full':
                result = (result.astype(np.float32) - output_zero) * output_scale
            scores[i] = result
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('출력 확률에 잘못된 값이 있습니다.')
        tolerance = len(labels) * output_scale + 1e-5 if name == 'int8_full' else 1e-5
        np.testing.assert_allclose(scores.sum(axis=1), 1, atol=tolerance)
        prediction = scores.argmax(axis=1)
        matrix = confusion_matrix(truth, prediction, labels=np.arange(len(labels)))
        accuracy = float(accuracy_score(truth, prediction) * 100)
        f1 = float(f1_score(truth, prediction, labels=np.arange(len(labels)), average='macro', zero_division=0) * 100)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        manual_f1 = np.divide(2 * matrix.diagonal(), denominator,
                              out=np.zeros(len(labels)), where=denominator != 0).mean() * 100
        np.testing.assert_allclose(accuracy, np.trace(matrix) / len(truth) * 100, atol=1e-12)
        np.testing.assert_allclose(f1, manual_f1, atol=1e-12)
        floating_ops = []
        if name == 'int8_full':
            floating_indices = {t['index'] for t in interpreter.get_tensor_details()
                                if np.issubdtype(t['dtype'], np.floating)}
            # LiteRT 2.3.0 diagnostic API: inspect graph operations, excluding delegate wrappers.
            for op in interpreter._get_ops_details():
                if op['op_name'] != 'DELEGATE' and any(int(i) in floating_indices
                        for i in [*op['inputs'], *op['outputs']]):
                    floating_ops.append({'operator': op['op_name'],
                                         'inputs': [int(i) for i in op['inputs']],
                                         'outputs': [int(i) for i in op['outputs']]})
        record = {
            'status': 'passed', 'bytes': path.stat().st_size, 'kib': path.stat().st_size / 1024,
            'sha256': digest(path), 'evaluated_windows': len(truth),
            'accuracy_percent': accuracy, 'macro_f1_percent': f1,
            'accuracy_difference_pp': accuracy - baseline['tflite_fp32']['accuracy_percent'],
            'macro_f1_difference_pp': f1 - baseline['tflite_fp32']['macro_f1_percent'],
            'correct_windows': int(np.trace(matrix)), 'confusion_matrix': matrix.tolist(),
            'prediction_agreement_with_fp32_percent': float(np.mean(prediction == baseline_prediction) * 100),
            'size_reduction_from_tflite_fp32_percent': (1 - path.stat().st_size / baseline['tflite_bytes']) * 100,
            'input_dtype': str(np.dtype(inp['dtype'])), 'output_dtype': str(np.dtype(out['dtype'])),
            'input_quantization': [float(input_scale), int(input_zero)],
            'output_quantization': [float(output_scale), int(output_zero)],
            'tensor_dtypes': dict(dtype_counts),
            'floating_tensor_count': sum(n for dtype, n in dtype_counts.items() if dtype.startswith('float')),
            'integer_only_achieved': not floating_indices if name == 'int8_full' else None,
            'floating_operators': floating_ops,
            'input_clipped_values': clipped, 'input_values': int(inputs.size),
        }
        return record, prediction

    report = {
        'platform': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ('tensorflow', 'keras', 'ai-edge-litert', 'numpy', 'scikit-learn')},
        'source_keras_sha256': original['sha256'], 'fp32_model_sha256': baseline['tflite_sha256'],
        'fp32_baseline': {'bytes': baseline['tflite_bytes'], 'kib': baseline['tflite_kib'], **baseline['tflite_fp32']},
        'preprocessing_artifact_sha256': preparation['artifact_sha256'], 'label_order': labels,
        'calibration': {'sampling': 'numpy.random.RandomState(42).choice, replace=False',
                        'windows': len(selected), 'source': 'X_train_s.npy',
                        'test_window_id_overlap': 0, 'indices_sha256': digest(output / 'calibration.csv')},
        'conversion': {'optimizations': ['DEFAULT'], 'select_tf_ops_enabled': False,
                       'fp16': 'supported_types=[float16]', 'int8_dynamic': 'no representative data',
                       'int8_full': 'TFLITE_BUILTINS_INT8; int8 input/output; representative training data'},
        'interpreter_threads': 1, 'training_executed': False, 'latency_measured': False,
        'matches_saved_model_training_statistics': preparation['matches_saved_model_training_statistics'],
        'variants': {},
    }
    predictions = {}
    model = keras.models.load_model(ROOT / model_config['local_path'],
        custom_objects={'AddPositionEmbedding': AddPositionEmbedding}, compile=False, safe_mode=True)
    with tempfile.TemporaryDirectory(prefix='wisdm-quantize-', dir=ROOT / 'models') as temporary:
        export = Path(temporary) / 'saved_model'
        model.export(str(export), format='tf_saved_model', verbose=False)
        for name, path in paths.items():
            print(f'변환·평가 시작: {name}', flush=True)
            try:
                converter = tf.lite.TFLiteConverter.from_saved_model(str(export))
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                if name == 'fp16':
                    converter.target_spec.supported_types = [tf.float16]
                elif name == 'int8_full':
                    converter.representative_dataset = representative_data
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
                path.write_bytes(converter.convert())
                report['variants'][name], predictions[name] = evaluate(path, name)
                print(json.dumps({'variant': name, **{k: report['variants'][name][k] for k in
                    ('kib', 'accuracy_percent', 'macro_f1_percent')}}, ensure_ascii=False), flush=True)
            except Exception as error:
                report['variants'][name] = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
                print(f'{name} 실패: {type(error).__name__}: {error}', flush=True)
            (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (output / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['test_index', 'window_id', 'true_label', 'fp32_label', *predictions])
        writer.writerows((i, int(test_ids[i]), int(truth[i]), int(baseline_prediction[i]),
                          *(int(predictions[name][i]) for name in predictions)) for i in range(len(truth)))
    report['predictions_sha256'] = digest(output / 'predictions.csv')
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0 if all(r['status'] == 'passed' for r in report['variants'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
