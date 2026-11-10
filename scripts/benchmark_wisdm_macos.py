"""Benchmark WISDM CPU inference in separate processes on Apple Silicon macOS."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def load_sources():
    prep = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text())
    folder = ROOT / prep['output_directory']
    preparation = json.loads((folder / 'preprocessing-summary.json').read_text())
    quant = json.loads((ROOT / 'results/2026-11-10-macos-wisdm-quantization/metrics.json').read_text())
    model = json.loads((ROOT / 'configs/wisdm-model.json').read_text())
    paths = {'keras_predict': ROOT / model['local_path'], 'keras_direct': ROOT / model['local_path']}
    paths.update({f'tflite_{name}': ROOT / f'models/wisdm_transformer_{name}.tflite'
                  for name in ('fp32', 'fp16', 'int8_dynamic', 'int8_full')})
    expected = {'keras_predict': quant['source_keras_sha256'], 'keras_direct': quant['source_keras_sha256'],
                'tflite_fp32': quant['fp32_model_sha256']}
    expected.update({f'tflite_{name}': r['sha256'] for name, r in quant['variants'].items()})
    return folder, preparation, quant, paths, expected


def worker(case, mode, output):
    import numpy as np
    from prepare_wisdm_data import digest
    folder, preparation, quant, paths, expected = load_sources()
    path = paths[case]
    if digest(path) != expected[case]:
        raise ValueError('모델 해시가 다릅니다.')
    config = json.loads((ROOT / 'configs/benchmark.json').read_text())
    count, warmup = config['measurement_iterations'], config['warmup_iterations']
    array = np.load(folder / 'X_test_s.npy', mmap_mode='r', allow_pickle=False)
    selected = np.random.RandomState(config['seed']).choice(len(array), count, replace=False)
    values = np.ascontiguousarray(array[selected], dtype=np.float32)
    del array
    samples = [values[i:i + 1] for i in range(count)]
    window_ids = np.load(folder / 'test_window_ids.npy', allow_pickle=False)[selected]
    if case.startswith('keras'):
        prior_path = ROOT / 'results/2026-11-09-macos-wisdm-tflite-fp32/predictions.csv'
        prior_key = 'keras_label'
    else:
        prior_path = ROOT / 'results/2026-11-10-macos-wisdm-quantization/predictions.csv'
        prior_key = 'fp32_label' if case == 'tflite_fp32' else case.removeprefix('tflite_')
    with prior_path.open() as stream:
        prior = list(csv.DictReader(stream))
    expected_labels = [int(prior[int(i)][prior_key]) for i in selected]
    runtime = {}
    if case.startswith('keras'):
        import tensorflow as tf
        import keras
        sys.path.insert(0, str(ROOT / 'src'))
        from har_quantization.layers import AddPositionEmbedding
        tf.config.set_visible_devices([], 'GPU')
        if mode == '1':
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        model = keras.models.load_model(path, custom_objects={'AddPositionEmbedding': AddPositionEmbedding},
                                        compile=False, safe_mode=True)
        runtime = {'tensorflow_intra_op': tf.config.threading.get_intra_op_parallelism_threads(),
                   'tensorflow_inter_op': tf.config.threading.get_inter_op_parallelism_threads()}
        if case == 'keras_predict':
            def invoke(sample):
                return model.predict(sample, batch_size=1, verbose=0)
        else:
            def invoke(sample):
                return model(sample, training=False).numpy()
    else:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(path), **({'num_threads': 1} if mode == '1' else {}))
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        runtime = {'interpreter_num_threads_requested': 1 if mode == '1' else None,
                   'delegate': 'default CPU delegate; XNNPACK creation confirmed in process log'}
        if case == 'tflite_int8_full':
            in_scale, in_zero = inp['quantization']
            out_scale, out_zero = out['quantization']
            def invoke(sample):
                x = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)
                interpreter.set_tensor(inp['index'], x)
                interpreter.invoke()
                return (interpreter.get_tensor(out['index']).astype(np.float32) - out_zero) * out_scale
        else:
            def invoke(sample):
                interpreter.set_tensor(inp['index'], sample)
                interpreter.invoke()
                return interpreter.get_tensor(out['index'])
    print(f'Warmup {warmup}; measure {count}: {case}/{mode}', flush=True)
    for i in range(warmup):
        invoke(samples[i % count])
    rows = []
    disagreement = 0
    for i, sample in enumerate(samples):
        start = time.perf_counter_ns()
        result = invoke(sample)
        duration = time.perf_counter_ns() - start
        if result.shape != (1, 6) or not np.isfinite(result).all():
            raise ValueError('잘못된 추론 출력이 있습니다.')
        label = int(result.argmax(axis=1)[0])
        disagreement += int(label != expected_labels[i])
        rows.append((i, int(selected[i]), int(window_ids[i]), duration, label))
    durations = np.array([row[3] for row in rows], dtype=np.float64) / 1e6
    raw = output / f'{case}-{mode}.csv'
    with raw.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['iteration', 'test_index', 'window_id', 'latency_ns', 'predicted_label'])
        writer.writerows(rows)
    report = {
        'variant': case, 'thread_mode': mode, 'iterations': count, 'warmup': warmup, 'batch_size': 1,
        'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95)),
        'mean_ms': float(durations.mean()), 'min_ms': float(durations.min()), 'max_ms': float(durations.max()),
        'native_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        'native_peak_rss_unit': 'bytes on Darwin', 'runtime_threads': runtime,
        'thread_environment': {key: os.environ.get(key) for key in THREAD_ENV},
        'tensorflow_imported': 'tensorflow' in sys.modules,
        'prediction_disagreements_with_prior_full_evaluation': disagreement,
        'model_bytes': path.stat().st_size, 'model_sha256': digest(path),
        'raw_csv': raw.name, 'raw_csv_sha256': digest(raw),
    }
    write_json(output / f'{case}-{mode}.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-case', choices=['keras_predict', 'keras_direct', 'tflite_fp32',
                                                'tflite_fp16', 'tflite_int8_dynamic', 'tflite_int8_full'])
    parser.add_argument('--worker-mode', choices=['1', 'default'])
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or sys.prefix == sys.base_prefix:
        raise ValueError('Mac ARM64 프로젝트 가상환경에서 실행해주세요.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.worker_case:
        worker(args.worker_case, args.worker_mode, output)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('기존 결과를 덮어쓰지 않습니다. 다른 --output 경로를 지정해주세요.')
    import numpy as np
    import psutil
    from prepare_wisdm_data import digest
    folder, preparation, quant, paths, expected = load_sources()
    for name, sha in preparation['artifact_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError(f'전처리 파일 해시가 다릅니다: {name}')
    for case, path in paths.items():
        if digest(path) != expected[case]:
            raise ValueError(f'모델 해시가 다릅니다: {case}')
    config = json.loads((ROOT / 'configs/benchmark.json').read_text())
    if config['batch_size'] != 1 or config['warmup_iterations'] != 50 or config['measurement_iterations'] != 1000:
        raise ValueError('예정한 배치·워밍업·측정 조건과 다릅니다.')
    output.mkdir(parents=True)
    power = subprocess.run(['/usr/bin/pmset', '-g', 'batt'], capture_output=True, text=True)
    # Record power status without battery hardware identifiers.
    power_lines = power.stdout.splitlines()
    power_summary = power_lines[0] if power_lines else 'unavailable'
    if len(power_lines) > 1 and '\t' in power_lines[1]:
        power_summary += '; ' + power_lines[1].split('\t', 1)[1]
    plan = [(case, str(mode)) for mode in config['threads'] for case in config['variants']]
    np.random.RandomState(config['seed']).shuffle(plan)
    environment = {
        'platform': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(),
        'logical_cpu_count': os.cpu_count(), 'power_at_start': power_summary,
        'packages': {p: importlib.metadata.version(p) for p in ('tensorflow', 'keras', 'ai-edge-litert', 'numpy', 'psutil')},
        'plan': [{'variant': c, 'thread_mode': m} for c, m in plan], 'seed': config['seed'],
        'input_sha256': preparation['artifact_sha256']['X_test_s.npy'],
        'note': 'Background applications and CPU frequency were not controlled; CPU brand/memory sysctl access was unavailable.',
    }
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    if ' @ ' in freeze or 'file://' in freeze:
        raise ValueError('의존성 목록에 로컬 경로가 있습니다. 공개 전 확인해주세요.')
    (output / 'requirements.txt').write_text(freeze)
    write_json(output / 'environment.json', environment)
    started = time.time()
    summaries = []
    for case, mode in plan:
        env = os.environ.copy()
        for key in THREAD_ENV:
            env.pop(key, None)
        if mode == '1':
            env.update({key: '1' for key in THREAD_ENV})
        env.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
                   KERAS_HOME=str(Path(sys.prefix) / '.keras'), MPLCONFIGDIR=str(Path(sys.prefix) / '.matplotlib'),
                   TF_CPP_MIN_LOG_LEVEL='2')
        print(f'측정 시작 {len(summaries) + 1}/{len(plan)}: {case}, threads={mode}', flush=True)
        peak_rss = peak_threads = 0
        with (output / f'{case}-{mode}.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                                        '--worker-case', case, '--worker-mode', mode], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            observed = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak_rss = max(peak_rss, observed.memory_info().rss)
                    peak_threads = max(peak_threads, observed.num_threads())
                except psutil.NoSuchProcess:
                    pass
                time.sleep(0.01)
            if process.returncode:
                raise RuntimeError(f'{case}/{mode} 실패. 해당 .log를 확인해주세요.')
        record_path = output / f'{case}-{mode}.json'
        record = json.loads(record_path.read_text())
        record.update(sampled_peak_rss_bytes=peak_rss, sampled_peak_os_threads=peak_threads)
        if case.startswith('tflite'):
            log_text = (output / f'{case}-{mode}.log').read_text()
            if 'Created TensorFlow Lite XNNPACK delegate for CPU' not in log_text:
                record['runtime_threads']['delegate'] = 'default CPU configuration; XNNPACK log not observed'
        write_json(record_path, record)
        summaries.append(record)
        write_json(output / 'summary.json', {'cases': summaries, 'elapsed_seconds': time.time() - started})
        print(f"완료: median={record['median_ms']:.6f} ms, p95={record['p95_ms']:.6f} ms, "
              f"peak RSS={record['native_peak_rss_bytes'] / 1024**2:.2f} MiB", flush=True)
    with (output / 'summary.csv').open('w', newline='') as stream:
        fields = ['variant', 'thread_mode', 'iterations', 'median_ms', 'p95_ms', 'mean_ms',
                  'native_peak_rss_bytes', 'sampled_peak_rss_bytes', 'sampled_peak_os_threads',
                  'prediction_disagreements_with_prior_full_evaluation', 'model_bytes']
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    write_json(output / 'summary.json', {
        'cases': summaries, 'elapsed_seconds': time.time() - started,
        'timing_scope': 'Input transfer/conversion, invocation, output materialization; INT8 quantize/dequantize included. Model loading/initialization and input file I/O excluded.',
        'memory_scope': 'Whole fresh child process peak RSS, including imports, model, selected input arrays, warmup and measurements. Not model-only memory.',
        'percentile_method': 'numpy percentile default linear',
        'accuracy_scope': '1000 benchmark outputs checked against prior full-evaluation predictions; not a new full accuracy evaluation.',
    })
    print(f'12개 조건 완료: {output}', flush=True)


if __name__ == '__main__':
    main()
