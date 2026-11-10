"""Convert/evaluate PAMAP2 TFLite FP32, FP16, dynamic INT8 and INT8-I/O."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from har_quantization.layers import AddPositionEmbedding
import keras
from ai_edge_litert.interpreter import Interpreter


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    model_config = json.loads((ROOT / 'configs/pamap2-model.json').read_text(encoding='utf-8'))
    model_path = ROOT / model_config['local_path']
    if digest(model_path) != model_config['sha256']:
        raise ValueError('PAMAP2 model hash mismatch')
    prep_dir = ROOT / 'data/processed/pamap2-author-code'
    prep = json.loads((prep_dir / 'preprocessing-summary.json').read_text(encoding='utf-8'))
    for name, expected in prep['artifact_sha256'].items():
        if digest(prep_dir / name) != expected:
            raise ValueError(f'Preprocessing artifact hash mismatch: {name}')
    train = np.load(prep_dir / 'X_train_s.npy', allow_pickle=False)
    test = np.load(prep_dir / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prep_dir / 'y_test.npy', allow_pickle=False)
    test_ids = np.load(prep_dir / 'test_window_ids.npy', allow_pickle=False)
    rng = np.random.RandomState(42)
    calibration_ids = rng.choice(len(train), min(300, len(train)), replace=False)
    output = ROOT / 'results/2026-11-10-windows-pamap2-quantization'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'calibration.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['train_index', 'window_id'])
        train_ids = np.load(prep_dir / 'train_window_ids.npy', allow_pickle=False)
        writer.writerows((int(index), int(train_ids[index])) for index in calibration_ids)
    model = keras.models.load_model(model_path, custom_objects={'AddPositionEmbedding': AddPositionEmbedding},
                                    compile=False, safe_mode=True)
    baseline = model(test, training=False).numpy()
    labels = len(prep['activity_ids'])
    results, predictions = {}, {'test_index': np.arange(len(truth)), 'window_id': test_ids, 'y_true': truth}
    model_paths = {key: ROOT / f'models/pamap2_transformer_{key}.tflite'
                   for key in ('fp32', 'fp16', 'int8_dynamic', 'int8_full')}

    def representative_data():
        for index in calibration_ids:
            yield [train[index:index + 1].astype(np.float32)]

    with tempfile.TemporaryDirectory(prefix='pamap2-export-', dir=ROOT / 'models') as tmp:
        export = Path(tmp) / 'saved_model'
        model.export(str(export), format='tf_saved_model', verbose=False)
        for variant, path in model_paths.items():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(export))
            converter.optimizations = [tf.lite.Optimize.DEFAULT] if variant != 'fp32' else []
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            if variant == 'fp16':
                converter.target_spec.supported_types = [tf.float16]
            elif variant == 'int8_full':
                converter.representative_dataset = representative_data
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            path.write_bytes(converter.convert())
            interpreter = Interpreter(model_path=str(path), num_threads=1)
            interpreter.allocate_tensors()
            inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            if list(inp['shape']) != [1, 200, 19] or list(out['shape']) != [1, labels]:
                raise ValueError(f'{variant}: wrong TFLite tensor shape')
            score = np.empty((len(test), labels), dtype=np.float32)
            for i, sample in enumerate(test):
                sample = sample[None, ...]
                if variant == 'int8_full':
                    scale, zero = inp['quantization']
                    qsample = np.clip(np.rint(sample / scale + zero), -128, 127).astype(np.int8)
                else:
                    qsample = sample.astype(np.float32)
                interpreter.set_tensor(inp['index'], qsample)
                interpreter.invoke()
                result = interpreter.get_tensor(out['index'])[0]
                if variant == 'int8_full':
                    scale, zero = out['quantization']
                    result = (result.astype(np.float32) - zero) * scale
                score[i] = result
            predicted = score.argmax(axis=1).astype(np.int32)
            matrix = confusion_matrix(truth, predicted, labels=np.arange(labels))
            accuracy = float(accuracy_score(truth, predicted) * 100)
            macro = float(f1_score(truth, predicted, average='macro', zero_division=0) * 100)
            entry = {
                'status': 'passed', 'bytes': path.stat().st_size, 'kib': path.stat().st_size / 1024,
                'sha256': digest(path), 'accuracy_percent': accuracy, 'macro_f1_percent': macro,
                'accuracy_difference_pp': accuracy - float(accuracy_score(truth, baseline.argmax(axis=1)) * 100),
                'macro_f1_difference_pp': macro - float(f1_score(truth, baseline.argmax(axis=1), average='macro', zero_division=0) * 100),
                'prediction_agreement_with_fp32_percent': float(np.mean(predicted == baseline.argmax(axis=1)) * 100),
                'input_dtype': np.dtype(inp['dtype']).name, 'output_dtype': np.dtype(out['dtype']).name,
                'input_quantization': list(inp['quantization']), 'output_quantization': list(out['quantization']),
                'integer_only_achieved': variant == 'int8_full' and all(
                    np.issubdtype(item['dtype'], np.integer) for item in interpreter.get_tensor_details()),
                'floating_operator_types': sorted({op['op_name'] for op in interpreter._get_ops_details()
                    if op['op_name'] != 'DELEGATE' and any(int(index) in {
                        tensor['index'] for tensor in interpreter.get_tensor_details()
                        if np.issubdtype(tensor['dtype'], np.floating)
                    } for index in [*op['inputs'], *op['outputs']])}) if variant == 'int8_full' else [],
                'confusion_matrix': matrix.tolist(),
            }
            results[variant] = entry
            predictions[variant] = predicted
            print(f"{variant}: size={entry['kib']:.1f} KiB acc={accuracy:.3f}% macroF1={macro:.3f}%", flush=True)
    pred_path = output / 'predictions.csv'
    with pred_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(predictions.keys())
        writer.writerows(zip(*predictions.values()))
    metrics = {
        'dataset': 'PAMAP2 Physical Activity Monitoring', 'model_sha256': model_config['sha256'],
        'model_parameter_count': int(model.count_params()),
        'preprocessing_summary_sha256': digest(prep_dir / 'preprocessing-summary.json'),
        'preprocessing_artifact_sha256': prep['artifact_sha256'],
        'evaluated_windows': len(truth), 'calibration': {'windows': len(calibration_ids), 'seed': 42,
                                                           'train_indices_sha256': digest(output / 'calibration.csv')},
        'variants': results, 'training_executed': False,
        'matches_saved_model_training_statistics': 'unverified',
        'limitation': 'INT8 input/output does not imply every graph operation is integer-only; window-level test split and scaler caveats are recorded in the evaluation report.',
        'predictions_sha256': digest(pred_path),
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
