"""Build a Windows WISDM comparison table, report, and figure."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(path):
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/2026-11-10-windows-wisdm-benchmark-final'))
    parser.add_argument('--evaluation', type=Path, default=Path('results/2026-11-10-windows-wisdm-fp32'))
    parser.add_argument('--quantization', type=Path, default=Path('results/2026-11-10-windows-wisdm-quantization'))
    args = parser.parse_args()
    output, evaluation_dir, quant_dir = map(resolve, (args.output, args.evaluation, args.quantization))
    eval_metrics = read_json(evaluation_dir / 'metrics.json')
    quant_metrics = read_json(quant_dir / 'metrics.json')
    model_config = read_json(ROOT / 'configs/wisdm-model.json')
    keras_bytes = (ROOT / model_config['local_path']).stat().st_size
    parameter_count = model_config['parameter_count']
    bench = {item['variant'] + ':' + item['thread_mode']: item
             for item in read_json(output / 'summary.json')['cases']}
    specs = [
        ('keras_predict', 'Keras predict()', eval_metrics['accuracy_percent'], eval_metrics['macro_f1_percent'],
         keras_bytes, '.keras'),
        ('keras_direct', 'Keras model(x)', eval_metrics['accuracy_percent'], eval_metrics['macro_f1_percent'],
         keras_bytes, '.keras'),
        ('tflite_fp32', 'TFLite FP32', quant_metrics['fp32_baseline']['accuracy_percent'],
         quant_metrics['fp32_baseline']['macro_f1_percent'], quant_metrics['fp32_baseline']['bytes'], '.tflite'),
    ]
    for key, label in [('fp16', 'TFLite FP16'), ('int8_dynamic', 'TFLite INT8 dynamic'),
                       ('int8_full', 'TFLite INT8 I/O (some FP32 ops)')]:
        item = quant_metrics['variants'][key]
        specs.append(('tflite_' + key, label, item['accuracy_percent'], item['macro_f1_percent'],
                      item['bytes'], '.tflite'))
    fp32_size = quant_metrics['fp32_baseline']['bytes']
    keras_size = keras_bytes
    rows = []
    for variant, label, accuracy, macro_f1, size_bytes, artifact_type in specs:
        default = bench[variant + ':default']
        one_thread = bench[variant + ':1']
        rows.append({
            'variant': variant, 'label': label, 'artifact_type': artifact_type,
            'parameter_count': parameter_count,
            'artifact_bytes': size_bytes, 'artifact_kib': size_bytes / 1024,
            'size_reduction_vs_keras_percent': 100 * (1 - size_bytes / keras_size),
            'size_reduction_vs_tflite_fp32_percent': 100 * (1 - size_bytes / fp32_size),
            'accuracy_percent': accuracy, 'macro_f1_percent': macro_f1,
            'accuracy_delta_vs_tflite_fp32_pp': accuracy - quant_metrics['fp32_baseline']['accuracy_percent'],
            'macro_f1_delta_vs_tflite_fp32_pp': macro_f1 - quant_metrics['fp32_baseline']['macro_f1_percent'],
            'default_median_ms': default['median_ms'], 'default_p95_ms': default['p95_ms'],
            'one_thread_median_ms': one_thread['median_ms'], 'one_thread_p95_ms': one_thread['p95_ms'],
            'default_peak_working_set_mib': default['peak_working_set_bytes'] / 1024**2,
            'one_thread_peak_working_set_mib': one_thread['peak_working_set_bytes'] / 1024**2,
            'benchmark_prediction_disagreements': default['prediction_disagreements_with_prior_full_evaluation'],
        })
    table_path = output / 'windows-results.csv'
    with table_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    md = [
        '# Windows WISDM 실험 결과', '',
        '2026-11-10 실행. 본 PC의 CPU 결과이며 다른 PC나 Windows 전체의 성능을 대표하지 않습니다.', '',
        '## 환경과 조건', '',
        '- Windows 11, AMD64, Python 3.12.10, 8 논리 CPU, 약 15.68 GiB RAM',
        '- TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0, NumPy 2.5.4',
        '- 실제 WISDM 테스트 입력, batch size 1, 워밍업 50회, 측정 1,000회',
        '- 1-thread와 기본 스레드 설정을 각각 새 프로세스에서 실행. 표의 지연시간은 기본 설정이며 두 설정 모두 CSV에 포함',
        '- 전원 프로필은 Samsung Mode. 백그라운드 부하·CPU 주파수·열 상태는 통제하지 않음',
        '- 지연시간에는 추론 호출과 텐서 입출력이 포함되며 모델 로드와 파일 I/O는 제외',
        '- 메모리는 런타임·모델·입력을 포함한 각 프로세스의 Windows peak working set', '',
        '## 정확도·크기·기본 스레드 성능', '',
        '| 실행 경로 | 파라미터 수 | 정확도 (%) | Macro-F1 (%) | 파일 크기 (KiB) | 중앙 지연 (ms) | p95 (ms) | Peak 메모리 (MiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        md.append(f"| {r['label']} | {r['parameter_count']:,} | {r['accuracy_percent']:.3f} | {r['macro_f1_percent']:.3f} | "
                  f"{r['artifact_kib']:.1f} | {r['default_median_ms']:.4f} | {r['default_p95_ms']:.4f} | "
                  f"{r['default_peak_working_set_mib']:.1f} |")
    md += [
        '', 'Keras 파일 크기는 `.keras`, 나머지는 각 `.tflite` 기준입니다. 압축률과 스레드별 전체 지표는 `windows-results.csv`를 참조합니다.',
        '', '![Windows WISDM accuracy, model size, latency and peak memory](windows-results.png)', '',
        '## 해석과 한계', '',
        '- FP32 Keras 평가는 테스트 5,358개 중 5,185개 정답, 정확도 96.771%, Macro-F1 95.458%입니다. 논문 표의 반올림 수치와 근접합니다.',
        '- 재학습하지 않았습니다. 원 학습 scaler 파일을 찾지 못해 표준화 통계가 원 모델 학습과 일치하는지는 확인하지 못했습니다.',
        '- 윈도우 단위 분할이라 같은 사용자와 인접 윈도우의 원본 센서 샘플이 학습·테스트에 함께 있습니다. 피험자 독립 일반화 성능으로 해석하지 않습니다.',
        '- INT8 I/O 변형은 입출력이 INT8이지만 일부 연산에 FP32 경로가 남아 완전 정수 연산 모델은 아닙니다.',
        '- 각 경로의 벤치마크 1,000개 예측 라벨은 전체 평가의 라벨과 모두 일치했습니다. 이는 별도의 전체 정확도 평가가 아닙니다.',
        '', '## 원시 기록과 재현', '',
        '- 환경: `environment.json`; 의존성: `requirements.txt`; 반복별 지연: `*-default.csv`, `*-1.csv`',
        '- 전체 요약: `summary.csv`, `summary.json`; 케이스별 로그와 JSON도 이 폴더에 보관',
        '- 표와 그림 재생성: `.venv\\Scripts\\python.exe scripts\\summarize_windows_results.py`',
        '- 전체 조건은 [실험 규약](../../docs/experiment-protocol.md), [모델 안내](../../docs/wisdm-model.md), [데이터 안내](../../docs/wisdm-data.md)을 참조',
    ]
    (output / 'windows-results.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    labels = [r['label'] for r in rows]
    colors = ['#777777', '#555555', '#2878B5', '#55A868', '#C44E52', '#8172B3']
    y = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.barh(y, [r['accuracy_percent'] for r in rows], color=colors, alpha=.85, label='Accuracy')
    ax.scatter([r['macro_f1_percent'] for r in rows], y, color='black', marker='|', s=180, label='Macro-F1')
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(94.5, 97.1)
    ax.set_xlabel('Score (%)'); ax.set_title('Test-set quality'); ax.legend(loc='lower right')
    ax = axes[0, 1]
    ax.barh(y, [r['default_median_ms'] for r in rows], color=colors, alpha=.85, label='Median')
    ax.scatter([r['default_p95_ms'] for r in rows], y, color='black', marker='|', s=180, label='p95')
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xscale('log')
    ax.set_xlabel('Latency (ms, log scale)'); ax.set_title('Default-thread inference'); ax.legend()
    ax = axes[1, 0]
    ax.barh(y, [r['artifact_kib'] for r in rows], color=colors, alpha=.85)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xscale('log')
    ax.set_xlabel('Artifact size (KiB, log scale)'); ax.set_title('Saved model size')
    ax = axes[1, 1]
    ax.barh(y, [r['default_peak_working_set_mib'] for r in rows], color=colors, alpha=.85)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel('Peak working set (MiB)'); ax.set_title('Default-thread process memory')
    for axis in axes.flat:
        axis.grid(axis='x', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle('WISDM Windows CPU comparison (batch 1, 1,000 timed inputs)', fontsize=14)
    fig.savefig(output / 'windows-results.png', dpi=180)
    plt.close(fig)
    print(f'Wrote {table_path}, {output / "windows-results.md"}, {output / "windows-results.png"}')


if __name__ == '__main__':
    main()
