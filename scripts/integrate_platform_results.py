"""Read pinned Git artifacts; verify and summarize existing HAR experiments.

No downloads, model loading, training, conversion, or inference are performed.
"""
import argparse
import csv
import hashlib
import io
import importlib.metadata
import platform
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('keras_predict', 'keras_direct', 'tflite_fp32', 'tflite_fp16',
            'tflite_int8_dynamic', 'tflite_int8_full')
LABELS = ('Keras predict()', 'Keras direct', 'TFLite FP32', 'FP16',
          'Dynamic INT8', 'INT8 I/O + FP32')
SOURCES = ROOT / 'configs/platform-result-sources.json'
OUTPUT = ROOT / 'results/platform-integration-draft'
ARTIFACTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, context):
    require(bool(np.isclose(actual, expected, rtol=0, atol=1e-9)),
            f'{context}: {actual} != {expected}')


def read(source, path):
    key = (source['commit'], path)
    if key not in ARTIFACTS:
        result = subprocess.run(['git', 'show', f'{key[0]}:{path}'], cwd=ROOT,
                                capture_output=True)
        if result.returncode:
            raise ValueError(f'Missing pinned artifact {key[0]}:{path}. '
                             'Fetch the source commits listed in the guide first.')
        ARTIFACTS[key] = result.stdout
    return ARTIFACTS[key]


def js(source, path):
    return json.loads(read(source, path).decode('utf-8-sig'))


def csv_rows(source, path):
    return list(csv.DictReader(io.StringIO(read(source, path).decode('utf-8-sig'))))


def check_hash(source, path, expected):
    require(hashlib.sha256(read(source, path)).hexdigest() == expected,
            f'Artifact hash mismatch: {path}')


def numbers(rows, key):
    return np.asarray([int(row[key]) for row in rows], dtype=np.int64)


def quality(truth, predicted, metric, classes):
    require(np.isin(truth, np.arange(classes)).all() and
            np.isin(predicted, np.arange(classes)).all(), 'Invalid class index')
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    acc = 100 * np.trace(matrix) / len(truth)
    denom = matrix.sum(0) + matrix.sum(1)
    macro = 100 * np.divide(2 * matrix.diagonal(), denom,
                           out=np.zeros(classes), where=denom != 0).mean()
    same(acc, metric['accuracy_percent'], 'Accuracy from predictions')
    same(macro, metric['macro_f1_percent'], 'Macro-F1 from predictions')
    require(np.array_equal(matrix, metric['confusion_matrix']), 'Confusion matrix mismatch')
    return float(acc), float(macro)


def collect(source):
    dataset, platform = source['dataset'], source['platform']
    classes = 6 if dataset == 'wisdm' else 18
    qdir, bdir = source['quantization'], source['benchmark']
    q = js(source, qdir + '/metrics.json')
    predictions = csv_rows(source, qdir + '/predictions.csv')
    check_hash(source, qdir + '/predictions.csv', q['predictions_sha256'])
    truth_key = 'y_true' if dataset == 'pamap2' and platform == 'windows' else 'true_label'
    truth, window_ids = numbers(predictions, truth_key), numbers(predictions, 'window_id')
    expected_count = 5358 if dataset == 'wisdm' else 494
    require(len(truth) == expected_count, 'Incomplete test evaluation')
    require(np.array_equal(numbers(predictions, 'test_index'), np.arange(len(truth))),
            'Evaluation test index order mismatch')
    if 'evaluation' in source:
        edir = source['evaluation']
        ev = js(source, edir + '/metrics.json')
        erows = csv_rows(source, edir + '/predictions.csv')
        check_hash(source, edir + '/predictions.csv',
                   ev.get('predictions_sha256', ev.get('prediction_sha256')))
        require(np.array_equal(window_ids, numbers(erows, 'window_id')),
                'Keras and quantization test windows differ')
        ekey = 'y_true' if dataset == 'pamap2' else 'true_label'
        require(np.array_equal(truth, numbers(erows, ekey)), 'Keras truth differs')
        pred_key = 'keras_label' if dataset == 'pamap2' else 'predicted_label'
        keras_pred = numbers(erows, pred_key)
        keras_metric, keras_sha = ev, ev['model_sha256']
    else:
        keras_metric, keras_pred = q['keras_reference'], numbers(predictions, 'keras_label')
        keras_sha = q['source_keras_sha256']
    require(q.get('source_keras_sha256', q.get('model_sha256')) == keras_sha,
            'Quantization source Keras model differs from evaluation')
    fp32_key = 'fp32' if dataset == 'pamap2' and platform == 'windows' else 'fp32_label'
    fp32 = q.get('fp32_baseline', q['variants'].get('fp32'))
    fp32_sha = q.get('fp32_model_sha256', fp32.get('sha256'))
    metadata = {'keras_predict': (keras_metric, keras_pred, keras_sha),
                'keras_direct': (keras_metric, keras_pred, keras_sha),
                'tflite_fp32': (fp32, numbers(predictions, fp32_key), fp32_sha)}
    for name in ('fp16', 'int8_dynamic', 'int8_full'):
        m = q['variants'][name]
        require(m.get('status') == 'passed', f'{name}: conversion not passed')
        metadata['tflite_' + name] = (m, numbers(predictions, name), m['sha256'])
    scores = {v: quality(truth, pred, m, classes)
              for v, (m, pred, _) in metadata.items()}
    cases = js(source, bdir + '/summary.json')['cases']
    environment = js(source, bdir + '/environment.json')
    require(len(cases) == 12 and {(c['variant'], c['thread_mode']) for c in cases}
            == {(v, t) for v in VARIANTS for t in ('1', 'default')}, 'Missing/duplicate case')
    rows, sequence = [], None
    for c in sorted(cases, key=lambda x: (x['thread_mode'], VARIANTS.index(x['variant']))):
        v, t = c['variant'], c['thread_mode']
        raw_path = bdir + '/' + c['raw_csv']
        raw = csv_rows(source, raw_path)
        check_hash(source, raw_path, c['raw_csv_sha256'])
        require(js(source, f'{bdir}/{v}-{t}.json') == c, 'Case and summary JSON differ')
        require(len(raw) == c['iterations'] == 1000 and c['warmup'] == 50
                and c['batch_size'] == 1, 'Benchmark protocol differs')
        require(np.array_equal(numbers(raw, 'iteration'), np.arange(1000)), 'Iteration order')
        indices = numbers(raw, 'test_index')
        require(((indices >= 0) & (indices < len(truth))).all(), 'Invalid sampled test index')
        if sequence is None:
            sequence = indices
        require(np.array_equal(sequence, indices), 'Input order differs within platform/dataset')
        require(np.array_equal(numbers(raw, 'window_id'), window_ids[indices]), 'Window ID differs')
        require(np.array_equal(numbers(raw, 'predicted_label'), metadata[v][1][indices]),
                'Benchmark prediction differs from full evaluation')
        require(c['prediction_disagreements_with_prior_full_evaluation'] == 0, 'Recorded disagreements')
        ns = numbers(raw, 'latency_ns')
        require((ns > 0).all(), 'Nonpositive latency')
        ms = ns / 1e6
        median, p95 = float(np.median(ms)), float(np.percentile(ms, 95))
        same(median, c['median_ms'], 'Median')
        same(p95, c['p95_ms'], 'p95')
        same(float(ms.mean()), c['mean_ms'], 'Mean')
        require(c['model_sha256'] == metadata[v][2], 'Benchmark model differs from evaluation')
        if v.startswith('tflite'):
            require(c['model_bytes'] == metadata[v][0]['bytes'], 'Model size differs')
        memkey = 'native_peak_rss_bytes' if platform == 'macos' else 'peak_working_set_bytes'
        acc, macro = scores[v]
        rows.append(dict(dataset=dataset, platform=platform, variant=v, thread_mode=t,
                         accuracy_percent=acc, macro_f1_percent=macro, test_windows=len(truth),
                         model_bytes=c['model_bytes'], model_kib=c['model_bytes'] / 1024,
                         median_ms=median, p95_ms=p95, peak_process_memory_mib=c[memkey] / 1024**2,
                         memory_metric=memkey, timed_calls=len(raw), warmup_calls=c['warmup'],
                         unique_timed_test_windows=len(np.unique(indices)),
                         model_sha256=c['model_sha256'], source_commit=source['commit'],
                         source_pr=source['pr'], raw_csv_path=raw_path, raw_csv_sha256=c['raw_csv_sha256']))
    hashes = q['preprocessing_artifact_sha256']
    details = dict(dataset=dataset, platform=platform, commit=source['commit'],
                   pr=source['pr'], test_windows=len(truth), unique_timed_test_windows=len(set(sequence)),
                   input_order_sha256=hashlib.sha256(np.asarray(sequence, dtype='<i8').tobytes()).hexdigest(),
                   input_order_encoding='test indices, little-endian int64',
                   evaluation_artifacts={k: hashes[k] for k in ('X_test_s.npy', 'y_test.npy', 'test_window_ids.npy')},
                   environment=environment)
    return rows, details, window_ids, truth


def effects(rows):
    lookup = {(r['dataset'], r['platform'], r['thread_mode'], r['variant']): r for r in rows}
    pairs = [('keras_api', 'keras_predict', 'keras_direct'),
             ('runtime', 'keras_direct', 'tflite_fp32')]
    pairs += [('quantization', 'tflite_fp32', v) for v in VARIANTS[3:]]
    result = []
    for dataset in ('wisdm', 'pamap2'):
        for platform in ('macos', 'windows'):
            for mode in ('1', 'default'):
                for category, base, target in pairs:
                    a, b = (lookup[dataset, platform, mode, v] for v in (base, target))
                    result.append(dict(dataset=dataset, platform=platform, thread_mode=mode,
                        effect=category, baseline=base, target=target,
                        baseline_median_ms=a['median_ms'], target_median_ms=b['median_ms'],
                        latency_reduction_percent=100 * (1 - b['median_ms'] / a['median_ms']),
                        speedup_factor=a['median_ms'] / b['median_ms'],
                        size_reduction_percent=100 * (1 - b['model_bytes'] / a['model_bytes']),
                        accuracy_delta_pp=b['accuracy_percent'] - a['accuracy_percent'],
                        macro_f1_delta_pp=b['macro_f1_percent'] - a['macro_f1_percent'],
                        peak_process_memory_reduction_percent=100 * (1 -
                            b['peak_process_memory_mib'] / a['peak_process_memory_mib'])))
    return result


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plots(out, rows, changes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    lookup = {(r['dataset'], r['platform'], r['thread_mode'], r['variant']): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for i, dataset in enumerate(('wisdm', 'pamap2')):
        for j, mode in enumerate(('1', 'default')):
            ax = axes[i, j]
            for offset, platform, color in [(-.18, 'macos', '#c66725'), (.18, 'windows', '#2269a1')]:
                vals = [lookup[dataset, platform, mode, v] for v in VARIANTS]
                pos = np.arange(6) + offset
                bars = ax.barh(pos, [r['median_ms'] for r in vals], .34, color=color, label=platform)
                ax.scatter([r['p95_ms'] for r in vals], pos, marker='|', color='black', s=50)
                for y, r in zip(pos, vals):
                    ax.text(.99, y, f"{r['median_ms']:.4g} / {r['p95_ms']:.4g}",
                            transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1))
            ax.set_yticks(np.arange(6), LABELS)
            ax.invert_yaxis()
            ax.set_xscale('log')
            ax.set_xlim(right=max(lookup[dataset, p, mode, v]['p95_ms']
                                  for p in ('macos', 'windows') for v in VARIANTS) * 100)
            ax.set_xlabel('Latency (ms, log scale); bars = median, black marks = p95')
            ax.set_title(f'{dataset.upper()} / threads={mode}')
            ax.text(.99, 1.01, 'median / p95 (ms)', transform=ax.transAxes, ha='right', fontsize=7)
            ax.grid(axis='x', alpha=.2)
            ax.legend(fontsize=8, loc='lower center')
    fig.suptitle('Observed Mac / Windows latency: matched thread setting, separate devices', fontsize=14)
    fig.text(.5, -.015, 'Batch 1, warmup 50, timed 1,000. PAMAP2 schedules differ: Mac cycles 494; Windows resamples (420 unique).', ha='center', fontsize=9)
    fig.savefig(out / 'latency.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    color_limit = max(abs(c['latency_reduction_percent']) for c in changes if c['effect'] == 'quantization')
    color_norm = TwoSlopeNorm(vmin=-np.ceil(color_limit / 100) * 100, vcenter=0, vmax=100)
    for ax, dataset in zip(axes, ('wisdm', 'pamap2')):
        labels, vals = [], []
        for platform in ('macos', 'windows'):
            for mode in ('1', 'default'):
                selected = [next(c for c in changes if c['dataset'] == dataset and c['platform'] == platform
                            and c['thread_mode'] == mode and c['target'] == v and c['effect'] == 'quantization')
                            for v in VARIANTS[3:]]
                vals.append([c['latency_reduction_percent'] for c in selected])
                labels.append(f'{platform} / {mode}')
        heatmap = ax.imshow(vals, cmap='RdYlGn', norm=color_norm, aspect='auto')
        for i in range(4):
            for j in range(3):
                ax.text(j, i, f'{vals[i][j]:+.1f}%', ha='center', va='center', fontsize=11)
        ax.set_yticks(range(4), labels)
        ax.set_xticks(range(3), ['FP16', 'Dynamic INT8', 'INT8 I/O + FP32'])
        ax.set_title(dataset.upper())
    fig.colorbar(heatmap, ax=axes, label='Latency reduction (%)', shrink=.8)
    fig.suptitle('Latency reduction vs TFLite FP32: positive = faster, negative = slower')
    fig.savefig(out / 'quantization-effects.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def report(out, rows, changes, config):
    lookup = {(r['dataset'], r['platform'], r['thread_mode'], r['variant']): r for r in rows}
    lines = ['# Mac·Windows 통합 결과 초안', '',
        '새 추론 실험이 아니라 검증된 PR #7·#14·#17의 고정 커밋에서 원시 CSV와 JSON을 읽어 만든 통합 분석입니다. '
        '선행 PR은 아직 main 반영 전이며, 이 문서는 최종 제출본이 아닙니다. 고정 커밋과 출처는 sources.json, 최신 승인/머지 상태는 각 PR에서 확인합니다.', '',
        '## 핵심 결과', '',
        '- 두 데이터셋·두 플랫폼의 1스레드/기본 설정 모두 동적 INT8이 TFLite 변형 중 중앙 지연시간이 가장 짧았습니다.',
        '- Keras에서 TFLite FP32로 바꿀 때의 변화가 양자화 변화보다 큽니다. 런타임·호출·텐서 처리 차이가 포함되므로 양자화 효과로 부르지 않습니다.',
        '- 파일이 작아져도 추론이 빨라진다고 보장할 수 없습니다. 크기 감소율과 지연시간 변화는 effects.csv에서 별도로 계산했습니다.',
        '- INT8 입출력 모델은 일부 FP32 연산을 포함하며, 이번 모든 조건에서 TFLite FP32보다 느렸습니다. 원인 분해/프로파일링은 하지 않았습니다.',
        '- FP16의 이득은 일정하지 않습니다. Windows WISDM은 기본 설정에서 약 18.2% 빨랐지만 연산 스레드 1에서는 약 206.2% 느렸습니다. Windows PAMAP2도 기본 설정에서 약 73.0% 빨랐지만 스레드 1에서는 약 1.7% 느렸습니다. 원인을 스레드나 OS 하나로 확정하지 않습니다.',
        '- 작은 정확도 변화는 평가 구간 몇 개의 차이입니다. 재학습이나 일반적인 정확도 향상을 입증한 결과가 아닙니다.', '',
        '## 정확도·크기', '',
        '전체 평가셋 기준이며 벤치마크 표본의 정확도가 아닙니다. Keras 두 호출 방식은 같은 저장 모델·전체 평가 값을 사용합니다.', '',
        '| 데이터 | 모델 | Mac 정확도/F1 (%) | Windows 정확도/F1 (%) | Mac/Windows 크기 (KiB) |',
        '| --- | --- | ---: | ---: | ---: |']
    for ds in ('wisdm', 'pamap2'):
        for v, label in zip(VARIANTS, LABELS):
            if v == 'keras_predict':
                continue
            a, b = (lookup[ds, p, 'default', v] for p in ('macos', 'windows'))
            lines.append(f"| {ds.upper()} | {label} | {a['accuracy_percent']:.3f} / {a['macro_f1_percent']:.3f} | "
                         f"{b['accuracy_percent']:.3f} / {b['macro_f1_percent']:.3f} | {a['model_kib']:.3f} / {b['model_kib']:.3f} |")
    for mode, title in [('default', '기본 설정'), ('1', '연산 스레드 1')]:
        lines += ['', f'## {title}의 지연시간·프로세스 메모리', '',
            'ms는 중앙값/p95, MiB는 프로세스 피크입니다. macOS는 native peak RSS, Windows는 peak working set이므로 OS 간 메모리 절대값/절감률을 직접 비교하지 않습니다.', '',
            '| 데이터 | 실행 경로 | Mac 중앙값/p95 (ms) | Windows 중앙값/p95 (ms) | Mac 피크 (MiB) | Windows 피크 (MiB) |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
        for ds in ('wisdm', 'pamap2'):
            for v, label in zip(VARIANTS, LABELS):
                a, b = (lookup[ds, p, mode, v] for p in ('macos', 'windows'))
                lines.append(f"| {ds.upper()} | {label} | {a['median_ms']:.6f} / {a['p95_ms']:.6f} | "
                             f"{b['median_ms']:.6f} / {b['p95_ms']:.6f} | {a['peak_process_memory_mib']:.2f} | {b['peak_process_memory_mib']:.2f} |")
    lines += ['', '![조건별 중앙값과 p95](latency.png)', '',
        '## 효과를 나눠 읽기', '',
        '감소율 = 100 × (1 − 대상 중앙값 / 기준 중앙값). 양수는 빨라짐, 음수는 느려짐입니다. '
        '원시 정밀도로 계산하고 표시할 때만 반올림했습니다. 전체 40개 비교는 effects.csv에 있습니다.', '',
        '| 데이터 | 플랫폼 | 스레드 | predict→direct 감소 | FP32→FP16 감소 | FP32→동적 INT8 감소 | FP32→INT8 입출력 감소 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for ds in ('wisdm', 'pamap2'):
        for p in ('macos', 'windows'):
            for mode in ('1', 'default'):
                filtered = [c for c in changes if (c['dataset'], c['platform'], c['thread_mode']) == (ds, p, mode)]
                vals = [filtered[i]['latency_reduction_percent'] for i in (0, 2, 3, 4)]
                lines.append(f"| {ds.upper()} | {p} | {mode} | " + ' | '.join(f'{v:+.1f}%' for v in vals) + ' |')
    lines += ['', '![FP32 기준 양자화 지연시간 감소율](quantization-effects.png)', '',
        '## 비교 조건과 한계', '',
        '- 모두 배치 1, 워밍업 50회, 측정 1,000회, 조건별 독립 프로세스입니다. 지연시간에는 호출·입출력 처리가 포함되고 모델 로드/파일 I/O는 제외됩니다. INT8 양자화·역양자화도 포함됩니다.',
        '- WISDM은 5,358개 중 비복원 1,000개, PAMAP2 전체 평가는 494개입니다. Mac PAMAP2 타이밍은 494개 전체 순환, Windows는 복원추출(고유 420개)입니다. 같은 스레드 모드라도 PAMAP2 입력 빈도/순서는 다릅니다.',
        '- 각 플랫폼·데이터셋의 12조건은 같은 입력 순서입니다. 두 플랫폼의 평가 입력/정답/분할 배열 해시는 일치하지만, 같은 전처리가 저자 모델 학습 때 쓰였다는 뜻은 아닙니다.',
        '- 서로 다른 CPU·장치·프로세스·세션입니다. 배경 부하·주파수·열 상태는 통제하지 않았습니다. ARM/x86 또는 OS 자체의 우열·인과 효과로 확대하지 않습니다.',
        '- 스레드 1은 연산 설정이며 전체 OS 스레드가 하나라는 뜻이 아닙니다. default의 실제 선택은 런타임에 맡깁니다. 실행 환경은 sources.json의 고정 커밋 environment 기록을 확인합니다.',
        '- 피크 메모리는 런타임·입력·모델을 포함하고, 로드/워밍업 중 피크도 포함될 수 있습니다. 모델 가중치 전용 RAM이나 추론 구간만의 피크가 아닙니다.',
        '- 윈도우 단위 무작위 분할로 사용자와 겹치는 센서 시점이 train/test에 함께 있습니다. PAMAP2는 파일 경계도 가로지를 수 있습니다. 피험자 독립 일반화 성능이 아닙니다.',
        '- 원 학습 scaler와 정확한 split 산출물이 없어 학습 통계 일치는 미확인입니다. PAMAP2 Keras 정확도/F1 97.368%/96.736%는 논문 보고값 98.480%/97.978%를 재현하지 못했습니다. 원인은 확정하지 않습니다.',
        '- 동일 크기라도 파일 SHA-256이 다를 수 있습니다. summary.csv에 모델 해시를 각각 남겼으며 변환 바이너리가 플랫폼 간 동일하다고 가정하지 않습니다.', '',
        '## 검증과 재생성', '',
        'validation.json은 48조건/48,000행의 해시·통계·입력 순서·예측 및 전체 평가 정확도/F1/혼동행렬 대조 결과입니다. '
        '이 검증은 제출된 결과의 정합성 검증이며 Windows 실행 과정을 직접 재실행한 것은 아닙니다.', '',
        'sources.json은 읽은 Git blob의 SHA-256과 영구 URL을 기록합니다. summary.csv는 48조건, effects.csv는 같은 플랫폼·스레드 안의 40개 비교입니다. '
        '실행 명령과 선행 PR 관계는 [통합 분석 안내](../../docs/platform-results-integration.md)에 있습니다.', '',
        '### 고정 출처', '']
    for s in config['sources']:
        url = f"https://github.com/{config['repository']}/tree/{s['commit']}/{s['benchmark']}"
        lines.append(f"- {s['dataset'].upper()} / {s['platform']}: [PR #{s['pr']}, {s['commit'][:8]}]({url})")
    (out / 'report.md').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    config = json.loads(SOURCES.read_text())
    rows, details, full_sets = [], [], {}
    for source in config['sources']:
        result, info, ids, truth = collect(source)
        rows.extend(result)
        details.append(info)
        full_sets[source['dataset'], source['platform']] = (ids, truth, info['evaluation_artifacts'])
    for ds in ('wisdm', 'pamap2'):
        a, b = full_sets[ds, 'macos'], full_sets[ds, 'windows']
        require(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), 'Cross-platform evaluation order/truth differs')
        require(a[2] == b[2], 'Cross-platform evaluation input hashes differ')
    require(len(rows) == 48, 'Expected 48 benchmark cases')
    for row in rows:
        peers = {r['variant']: r for r in rows if (r['dataset'], r['platform'], r['thread_mode'])
                 == (row['dataset'], row['platform'], row['thread_mode'])}
        row['size_reduction_vs_keras_percent'] = 100 * (1 - row['model_bytes'] / peers['keras_direct']['model_bytes'])
        row['size_reduction_vs_tflite_fp32_percent'] = 100 * (1 - row['model_bytes'] / peers['tflite_fp32']['model_bytes'])
    changes = effects(rows)
    require(len(changes) == 40, 'Expected 40 within-platform comparisons')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; choose a fresh --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'summary.csv', rows)
    write_csv(out / 'effects.csv', changes)
    artifacts = [dict(commit=sha, path=path, sha256=hashlib.sha256(data).hexdigest(),
                      url=f"https://github.com/{config['repository']}/blob/{sha}/{path}")
                 for (sha, path), data in sorted(ARTIFACTS.items())]
    write_json(out / 'sources.json', dict(config=config, measurements=details, artifacts=artifacts))
    write_json(out / 'validation.json', dict(status='passed', benchmark_cases=48, benchmark_rows=48000,
        comparisons=40, source_blob_count=len(artifacts), retraining=False, inference_executed=False,
        checks=['full-evaluation CSV hashes, accuracy, Macro-F1 and confusion matrices',
                '48 raw latency CSV hashes, positive times, iterations, warmup and batch size',
                'raw median, linear p95 and mean match summary and per-case JSON',
                'same sample order for all variants/thread modes within each dataset/platform',
                'all 48000 benchmark labels match full-evaluation predictions',
                'benchmark model hashes and TFLite sizes match evaluation metadata',
                'cross-platform evaluation array hashes, window IDs and truth match'],
        generation_packages=dict(python=platform.python_version(), numpy=np.__version__,
            matplotlib=importlib.metadata.version('matplotlib'))))
    plots(out, rows, changes)
    report(out, rows, changes, config)
    print(f'Verified 48 conditions / 48000 measurements; wrote {out}')


if __name__ == '__main__':
    main()
