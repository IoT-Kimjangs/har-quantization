"""Create the WISDM INT8 model with float32 input/output for the cause analysis.

The model uses the same converter settings and the same 300 calibration windows as the
existing INT8 input/output model, except that the model input and output stay float32.
The existing INT8 input/output model is converted again in a temporary folder to check
whether conversion is reproducible on this machine.
"""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from prepare_wisdm_data import digest

ROOT = Path(__file__).resolve().parents[1]
NAME = 'int8_float_io'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quant-result', type=Path,
                        default=Path('results/2026-11-10-macos-wisdm-quantization'))
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    quant_dir = args.quant_result if args.quant_result.is_absolute() else ROOT / args.quant_result
    model_path = ROOT / f'models/wisdm_transformer_{NAME}.tflite'
    if (output.exists() and any(output.iterdir())) or model_path.exists():
        raise ValueError('기존 결과나 모델이 있습니다. 다른 --output 경로를 쓰거나 기존 파일을 먼저 확인해주세요.')
    config = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text(encoding='utf-8'))
    model_config = json.loads((ROOT / 'configs/wisdm-model.json').read_text(encoding='utf-8'))
    labels = json.loads((ROOT / 'configs/wisdm-data.json').read_text(encoding='utf-8'))['label_order']
    prepared = ROOT / config['output_directory']
    preparation = json.loads((prepared / 'preprocessing-summary.json').read_text(encoding='utf-8'))
    for name, expected in preparation['artifact_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'전처리 산출물 해시가 다릅니다: {name}')
    quant = json.loads((quant_dir / 'metrics.json').read_text(encoding='utf-8'))
    keras_path = ROOT / model_config['local_path']
    if digest(keras_path) != quant['source_keras_sha256']:
        raise ValueError('원본 Keras 모델 해시가 양자화 기록과 다릅니다.')
    if digest(quant_dir / 'predictions.csv') != quant['predictions_sha256']:
        raise ValueError('기존 양자화 예측 기록 해시가 다릅니다.')

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
    from sklearn.metrics import accuracy_score, f1_score
    from har_quantization.layers import AddPositionEmbedding
    tf.config.set_visible_devices([], 'GPU')

    train = np.load(prepared / 'X_train_s.npy', allow_pickle=False)
    inputs = np.load(prepared / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prepared / 'y_test.npy', allow_pickle=False)
    train_ids = np.load(prepared / 'train_window_ids.npy', allow_pickle=False)
    selected = np.random.RandomState(42).choice(len(train), min(300, len(train)), replace=False)
    with (quant_dir / 'calibration.csv').open(newline='', encoding='utf-8') as stream:
        recorded = [(int(r['train_index']), int(r['window_id'])) for r in csv.DictReader(stream)]
    if recorded != [(int(i), int(train_ids[i])) for i in selected]:
        raise ValueError('보정 구간이 기존 INT8 입출력 변환 기록과 다릅니다.')
    with (quant_dir / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        prior = list(csv.DictReader(stream))
    int8_io_prediction = np.array([int(r['int8_full']) for r in prior])
    fp32_prediction = np.array([int(r['fp32_label']) for r in prior])

    def representative_data():
        for i in selected:
            yield [train[i:i + 1].astype(np.float32)]

    model = keras.models.load_model(keras_path, custom_objects={'AddPositionEmbedding': AddPositionEmbedding},
                                    compile=False, safe_mode=True)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wisdm-cause-', dir=ROOT / 'models') as temporary:
        export = Path(temporary) / 'saved_model'
        model.export(str(export), format='tf_saved_model', verbose=False)

        def convert(int8_io):
            converter = tf.lite.TFLiteConverter.from_saved_model(str(export))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if int8_io:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            return converter.convert()

        reconverted = Path(temporary) / 'int8_full_reconverted.tflite'
        reconverted.write_bytes(convert(True))
        reconverted_sha = digest(reconverted)
        model_path.write_bytes(convert(False))

    interpreter = Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if inp['dtype'] != np.float32 or out['dtype'] != np.float32:
        raise ValueError('입출력이 float32가 아닙니다.')
    if list(inp['shape']) != [1, 80, 3] or list(out['shape']) != [1, len(labels)]:
        raise ValueError('입출력 형태가 다릅니다.')
    tensor_dtypes = Counter(np.dtype(t['dtype']).name for t in interpreter.get_tensor_details())
    ops = Counter(op['op_name'] for op in interpreter._get_ops_details())
    scores = np.empty((len(inputs), len(labels)), dtype=np.float32)
    for i in range(len(inputs)):
        interpreter.set_tensor(inp['index'], inputs[i:i + 1])
        interpreter.invoke()
        scores[i] = interpreter.get_tensor(out['index'])[0]
    if not np.isfinite(scores).all():
        raise ValueError('출력에 비유한 값이 있습니다.')
    prediction = scores.argmax(axis=1)
    report = {
        'variant': NAME,
        'purpose': 'Cause analysis: same INT8 conversion as int8_full but float32 model input/output.',
        'platform': platform.platform(), 'python': platform.python_version(),
        'tensorflow': tf.__version__,
        'source_keras_sha256': quant['source_keras_sha256'],
        'calibration': {'windows': len(selected), 'matches_existing_int8_full_record': True},
        'conversion': {'optimizations': ['DEFAULT'], 'supported_ops': ['TFLITE_BUILTINS_INT8'],
                       'representative_data': 'same 300 training windows as int8_full',
                       'inference_input_type': 'float32 (default)', 'inference_output_type': 'float32 (default)'},
        'int8_full_reconversion_sha256': reconverted_sha,
        'int8_full_recorded_sha256': quant['variants']['int8_full']['sha256'],
        'int8_full_reconversion_identical': reconverted_sha == quant['variants']['int8_full']['sha256'],
        'model_bytes': model_path.stat().st_size, 'kib': model_path.stat().st_size / 1024,
        'sha256': digest(model_path),
        'input': {'dtype': 'float32', 'shape': [int(v) for v in inp['shape']]},
        'output': {'dtype': 'float32', 'shape': [int(v) for v in out['shape']]},
        'tensor_dtype_counts': dict(tensor_dtypes), 'op_counts': dict(sorted(ops.items())),
        'test_windows': len(truth),
        'accuracy_percent': float(accuracy_score(truth, prediction) * 100),
        'macro_f1_percent': float(f1_score(truth, prediction, labels=np.arange(len(labels)),
                                           average='macro', zero_division=0) * 100),
        'correct': int((prediction == truth).sum()),
        'agreement_with_int8_full_percent': float((prediction == int8_io_prediction).mean() * 100),
        'agreement_with_fp32_percent': float((prediction == fp32_prediction).mean() * 100),
    }
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['test_index', 'true_label', NAME])
        writer.writerows((i, int(truth[i]), int(prediction[i])) for i in range(len(truth)))
    report['predictions_sha256'] = digest(output / 'predictions.csv')
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('kib', 'accuracy_percent', 'correct', 'agreement_with_int8_full_percent',
                                             'int8_full_reconversion_identical')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
