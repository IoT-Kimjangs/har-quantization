"""Cause analysis benchmark for WISDM TFLite models (Mac ARM64 or Windows x86-64).

Experiment 1: split each call into input conversion, set+invoke, and output read/restore,
and compare INT8 input/output with an INT8 model that keeps float32 input/output.
Experiment 2: run every model with the default CPU delegate (XNNPACK) and with default
delegates disabled, and record how the graph is split into delegate partitions.
Each condition runs in a fresh process and is repeated. Batch 1, 50 warmup, 1,000 timed calls.
"""
import argparse
from collections import Counter
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')
VARIANTS = ('fp32', 'fp16', 'int8_dynamic', 'int8_full', 'int8_float_io')
RESOLVERS = ('default', 'no_default_delegates')
THREADS = ('1', 'default')


def digest(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def resolve(path):
    return path if path.is_absolute() else ROOT / path


def load_sources(quant_dir, float_io_dir):
    prep = json.loads((ROOT / 'configs/wisdm-preprocessing.json').read_text(encoding='utf-8'))
    folder = ROOT / prep['output_directory']
    preparation = json.loads((folder / 'preprocessing-summary.json').read_text(encoding='utf-8'))
    quant = json.loads((quant_dir / 'metrics.json').read_text(encoding='utf-8'))
    float_io = json.loads((float_io_dir / 'metrics.json').read_text(encoding='utf-8'))
    paths = {name: ROOT / f'models/wisdm_transformer_{name}.tflite' for name in VARIANTS}
    expected = {'fp32': quant['fp32_model_sha256'], 'int8_float_io': float_io['sha256']}
    expected.update({name: record['sha256'] for name, record in quant['variants'].items()})
    return folder, preparation, paths, expected


def worker(variant, resolver, mode, run, output, quant_dir, float_io_dir):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    folder, preparation, paths, expected = load_sources(quant_dir, float_io_dir)
    path = paths[variant]
    if digest(path) != expected[variant]:
        raise ValueError(f'{variant}: 모델 해시가 기록과 다릅니다.')
    config = json.loads((ROOT / 'configs/benchmark.json').read_text(encoding='utf-8'))
    count, warmup = config['measurement_iterations'], config['warmup_iterations']
    array = np.load(folder / 'X_test_s.npy', mmap_mode='r', allow_pickle=False)
    selected = np.random.RandomState(config['seed']).choice(len(array), count, replace=False)
    values = np.ascontiguousarray(array[selected], dtype=np.float32)
    del array
    samples = [values[i:i + 1] for i in range(count)]
    if variant == 'int8_float_io':
        prior_path, key = float_io_dir / 'predictions.csv', 'int8_float_io'
    else:
        prior_path, key = quant_dir / 'predictions.csv', 'fp32_label' if variant == 'fp32' else variant
    with prior_path.open(newline='', encoding='utf-8') as stream:
        prior = list(csv.DictReader(stream))
    expected_labels = [int(prior[int(i)][key]) for i in selected]

    options = {}
    if mode == '1':
        options['num_threads'] = 1
    if resolver == 'no_default_delegates':
        options['experimental_op_resolver_type'] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    interpreter = Interpreter(model_path=str(path), **options)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    ops = Counter(op['op_name'] for op in interpreter._get_ops_details())
    int8_input = inp['dtype'] == np.int8
    in_scale, in_zero = inp['quantization']
    out_scale, out_zero = out['quantization']
    clock = time.perf_counter_ns

    def call(sample):
        t0 = clock()
        x = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8) if int8_input else sample
        t1 = clock()
        interpreter.set_tensor(inp['index'], x)
        interpreter.invoke()
        t2 = clock()
        y = interpreter.get_tensor(out['index'])
        if int8_input:
            y = (y.astype(np.float32) - out_zero) * out_scale
        t3 = clock()
        return y, t1 - t0, t2 - t1, t3 - t2, t3 - t0

    print(f'Warmup {warmup}; measure {count}: {variant}/{resolver}/{mode}/run{run}', flush=True)
    for i in range(warmup):
        call(samples[i % count])
    rows, disagreement = [], 0
    for i, sample in enumerate(samples):
        result, q_ns, invoke_ns, out_ns, total_ns = call(sample)
        if result.shape != (1, 6) or not np.isfinite(result).all():
            raise ValueError('잘못된 추론 출력이 있습니다.')
        label = int(result.argmax(axis=1)[0])
        disagreement += int(label != expected_labels[i])
        rows.append((i, int(selected[i]), q_ns, invoke_ns, out_ns, total_ns, label))
    stem = f'{variant}-{resolver}-{mode}-run{run}'
    with (output / f'{stem}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['iteration', 'test_index', 'input_convert_ns', 'set_invoke_ns', 'output_ns',
                         'total_ns', 'predicted_label'])
        writer.writerows(rows)
    arr = np.array([r[2:6] for r in rows], dtype=np.float64) / 1e6
    med = np.median(arr, axis=0)
    record = {
        'variant': variant, 'resolver': resolver, 'thread_mode': mode, 'run': run,
        'iterations': count, 'warmup': warmup, 'batch_size': 1,
        'input_convert_median_ms': float(med[0]), 'set_invoke_median_ms': float(med[1]),
        'output_median_ms': float(med[2]), 'total_median_ms': float(med[3]),
        'total_p95_ms': float(np.percentile(arr[:, 3], 95)),
        'input_dtype': np.dtype(inp['dtype']).name, 'output_dtype': np.dtype(out['dtype']).name,
        'delegate_partitions': int(ops.get('DELEGATE', 0)),
        'model_op_count': int(sum(v for k, v in ops.items() if k != 'DELEGATE')),
        'op_counts_after_allocate': dict(sorted(ops.items())),
        'interpreter_options': {'num_threads': options.get('num_threads'),
                                'op_resolver': 'BUILTIN_WITHOUT_DEFAULT_DELEGATES' if resolver != 'default' else 'AUTO'},
        'prediction_disagreements_with_full_evaluation': disagreement,
        'model_sha256': expected[variant], 'raw_csv': f'{stem}.csv', 'raw_csv_sha256': digest(output / f'{stem}.csv'),
    }
    write_json(output / f'{stem}.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quant-result', type=Path, default=Path('results/2026-11-10-macos-wisdm-quantization'))
    parser.add_argument('--float-io-result', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', nargs=4, metavar=('VARIANT', 'RESOLVER', 'THREADS', 'RUN'))
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    output, quant_dir, float_io_dir = resolve(args.output), resolve(args.quant_result), resolve(args.float_io_result)
    if args.worker:
        variant, resolver, mode, run = args.worker
        worker(variant, resolver, mode, int(run), output, quant_dir, float_io_dir)
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('기존 결과를 덮어쓰지 않습니다. 다른 --output 경로를 지정해주세요.')
    import numpy as np
    folder, preparation, paths, expected = load_sources(quant_dir, float_io_dir)
    for name, sha in preparation['artifact_sha256'].items():
        if digest(folder / name) != sha:
            raise ValueError(f'전처리 파일 해시가 다릅니다: {name}')
    for name, path in paths.items():
        if digest(path) != expected[name]:
            raise ValueError(f'모델 해시가 다릅니다: {name}')
    config = json.loads((ROOT / 'configs/benchmark.json').read_text(encoding='utf-8'))
    if config['batch_size'] != 1 or config['warmup_iterations'] != 50 or config['measurement_iterations'] != 1000:
        raise ValueError('예정한 배치·워밍업·측정 조건과 다릅니다.')
    plan = [(v, r, m, k) for k in range(1, args.repeats + 1) for v in VARIANTS for r in RESOLVERS for m in THREADS]
    np.random.RandomState(config['seed']).shuffle(plan)
    output.mkdir(parents=True)
    write_json(output / 'environment.json', {
        'platform': platform.platform(), 'architecture': platform.machine(), 'python': platform.python_version(),
        'logical_cpu_count': os.cpu_count(),
        'packages': {p: importlib.metadata.version(p) for p in ('ai-edge-litert', 'numpy')},
        'plan': [{'variant': v, 'resolver': r, 'thread_mode': m, 'run': k} for v, r, m, k in plan],
        'quant_result': str(quant_dir.relative_to(ROOT)), 'float_io_result': str(float_io_dir.relative_to(ROOT)),
        'note': 'Background applications and CPU frequency were not controlled. Disabling default delegates is a diagnostic condition, not a recommended setting.',
    })
    records, started = [], time.time()
    for n, (v, r, m, k) in enumerate(plan, 1):
        env = os.environ.copy()
        for key in THREAD_ENV:
            env.pop(key, None)
        if m == '1':
            env.update({key: '1' for key in THREAD_ENV})
        env['TF_CPP_MIN_LOG_LEVEL'] = '0'
        stem = f'{v}-{r}-{m}-run{k}'
        print(f'[{n}/{len(plan)}] {stem}', flush=True)
        with (output / f'{stem}.log').open('w', encoding='utf-8') as log:
            code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                                    '--quant-result', str(quant_dir), '--float-io-result', str(float_io_dir),
                                    '--worker', v, r, m, str(k)], env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f'{stem} 실패. 해당 .log를 확인해주세요.')
        record = json.loads((output / f'{stem}.json').read_text(encoding='utf-8'))
        record['xnnpack_log_observed'] = 'XNNPACK delegate' in (output / f'{stem}.log').read_text(encoding='utf-8')
        write_json(output / f'{stem}.json', record)
        records.append(record)
    fields = ['variant', 'resolver', 'thread_mode', 'run', 'input_convert_median_ms', 'set_invoke_median_ms',
              'output_median_ms', 'total_median_ms', 'total_p95_ms', 'delegate_partitions', 'model_op_count',
              'xnnpack_log_observed', 'prediction_disagreements_with_full_evaluation']
    records.sort(key=lambda x: (VARIANTS.index(x['variant']), x['resolver'], x['thread_mode'], x['run']))
    with (output / 'runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    groups = {}
    for x in records:
        groups.setdefault((x['variant'], x['resolver'], x['thread_mode']), []).append(x)
    summary = []
    for (v, r, m), xs in groups.items():
        med = lambda key: float(np.median([x[key] for x in xs]))
        summary.append({'variant': v, 'resolver': r, 'thread_mode': m, 'runs': len(xs),
                        'total_median_ms': med('total_median_ms'),
                        'total_median_ms_min_run': min(x['total_median_ms'] for x in xs),
                        'total_median_ms_max_run': max(x['total_median_ms'] for x in xs),
                        'input_convert_median_ms': med('input_convert_median_ms'),
                        'set_invoke_median_ms': med('set_invoke_median_ms'), 'output_median_ms': med('output_median_ms'),
                        'delegate_partitions': xs[0]['delegate_partitions'], 'model_op_count': xs[0]['model_op_count'],
                        'prediction_disagreements': sum(x['prediction_disagreements_with_full_evaluation'] for x in xs)})
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)
    write_json(output / 'summary.json', {
        'conditions': summary, 'elapsed_seconds': time.time() - started,
        'timing_scope': 'Per call: input conversion (INT8 input only), set_tensor+invoke, output read and dequantize (INT8 output only). Model loading excluded.',
        'aggregation': 'Median over runs of the per-run median of 1,000 calls.',
    })
    print(f'{len(plan)}회 실행 완료: {output}', flush=True)


if __name__ == '__main__':
    main()
