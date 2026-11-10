"""Create the PAMAP2 Windows comparison table, report and figure."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(path):
    return path if path.is_absolute() else ROOT/path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('results/2026-11-10-windows-pamap2-benchmark-final'))
    ap.add_argument('--evaluation',type=Path,default=Path('results/2026-11-10-windows-pamap2-fp32'))
    ap.add_argument('--quantization',type=Path,default=Path('results/2026-11-10-windows-pamap2-quantization'))
    args=ap.parse_args(); out,evdir,qdir=map(resolve,(args.output,args.evaluation,args.quantization))
    ev=load(evdir/'metrics.json'); q=load(qdir/'metrics.json'); cfg=load(ROOT/'configs/pamap2-model.json')
    runs={x['variant']+':'+x['thread_mode']:x for x in load(out/'summary.json')['cases']}
    specs=[('keras_predict','Keras predict()',ev['accuracy_percent'],ev['macro_f1_percent'],cfg['bytes'],'.keras'),
           ('keras_direct','Keras model(x)',ev['accuracy_percent'],ev['macro_f1_percent'],cfg['bytes'],'.keras')]
    for key,label in [('fp32','TFLite FP32'),('fp16','TFLite FP16'),('int8_dynamic','TFLite INT8 dynamic'),('int8_full','TFLite INT8 I/O (some FP32 ops)')]:
        a=q['variants'][key]; specs.append(('tflite_'+key,label,a['accuracy_percent'],a['macro_f1_percent'],a['bytes'],'.tflite'))
    base=q['variants']['fp32']['bytes']; keras_bytes=cfg['bytes']; rows=[]
    for name,label,acc,f1,size,ext in specs:
        d=runs[name+':default']; one=runs[name+':1']
        rows.append({'variant':name,'label':label,'artifact_type':ext,'parameter_count':cfg['parameter_count'],'artifact_bytes':size,'artifact_kib':size/1024,
          'size_reduction_vs_keras_percent':100*(1-size/keras_bytes),'size_reduction_vs_tflite_fp32_percent':100*(1-size/base),
          'accuracy_percent':acc,'macro_f1_percent':f1,'accuracy_delta_vs_tflite_fp32_pp':acc-q['variants']['fp32']['accuracy_percent'],
          'macro_f1_delta_vs_tflite_fp32_pp':f1-q['variants']['fp32']['macro_f1_percent'],
          'default_median_ms':d['median_ms'],'default_p95_ms':d['p95_ms'],'one_thread_median_ms':one['median_ms'],'one_thread_p95_ms':one['p95_ms'],
          'default_peak_working_set_mib':d['peak_working_set_bytes']/1024**2,'one_thread_peak_working_set_mib':one['peak_working_set_bytes']/1024**2,
          'benchmark_prediction_disagreements':d['prediction_disagreements_with_prior_full_evaluation']})
    with (out/'pamap2-windows-results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    env=load(out/'environment.json')
    md=['# Windows PAMAP2 실험 결과','',
        '2026-11-10 실행. 본 Windows PC의 CPU 결과이며 다른 장치나 Windows 전체 성능을 대표하지 않습니다.','',
        '## 환경과 측정 조건','',
        f"- Windows AMD64, Python {env['python']}, CPU `{env['cpu']}`, {env['logical_cpu_count']} 논리 CPU, RAM {env['ram_gb']} GiB",
        '- TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0',
        '- batch size 1, 워밍업 50회, 측정 1,000회. 1-thread와 기본 스레드를 독립 프로세스에서 각각 실행',
        '- 테스트 분할은 494개 윈도우이므로 벤치마크 1,000개 입력은 seed 42로 복원추출',
        '- 전원 프로필은 Samsung Mode. 백그라운드 부하·CPU 주파수·온도 상태는 통제하지 않음',
        '- 지연시간은 추론 호출과 텐서 입출력을 포함하고 모델 로드·파일 I/O는 제외. INT8의 입력 양자화·출력 역양자화는 포함',
        '- peak 메모리는 모델·런타임·입력 배열을 포함한 독립 워커의 Windows peak working set','',
        '## 평가·크기·기본 스레드 성능','','| 실행 경로 | 파라미터 수 | 정확도 (%) | Macro-F1 (%) | 파일 크기 (KiB) | 중앙 지연 (ms) | p95 (ms) | Peak 메모리 (MiB) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['label']} | {r['parameter_count']:,} | {r['accuracy_percent']:.3f} | {r['macro_f1_percent']:.3f} | {r['artifact_kib']:.1f} | {r['default_median_ms']:.4f} | {r['default_p95_ms']:.4f} | {r['default_peak_working_set_mib']:.1f} |")
    md += ['', '크기는 Keras `.keras` 또는 각 TFLite `.tflite` 모델 파일 기준입니다. 두 기준별 크기 감소와 1-thread 수치는 `pamap2-windows-results.csv`에 있습니다.',
      '', '![Windows PAMAP2 model quality, size, latency and memory](pamap2-windows-results.png)','',
      '## 해석과 한계','',
      f"- 저장된 Keras 기준 모델은 테스트 {ev['evaluated_windows']}개에서 정확도 {ev['accuracy_percent']:.3f}%, Macro-F1 {ev['macro_f1_percent']:.3f}%를 기록했습니다. 논문의 98.48%는 다른 평가 설정의 참고값이며 직접 동등 비교할 수 없습니다.",
      '- 저자 노트북의 recursive file loading을 재현해 Protocol과 Optional의 14개 파일을 정렬·연결한 다음 결측·무한대와 activity 0을 제거하고 윈도우화했습니다. 파일/피험자 경계를 보존하지 않아 일부 윈도우가 경계를 넘을 수 있습니다.',
      '- 저자 노트북처럼 윈도우 단위 stratified random split을 사용했습니다. 겹치는 샘플과 같은 피험자가 train/test 양쪽에 포함될 수 있어 피험자 독립 성능이 아닙니다.',
      '- 원 학습 scaler 아티팩트를 확보하지 못해 저장 모델 학습 당시 표준화 통계와 같은지 검증하지 못했습니다. 재학습은 하지 않았습니다.',
      '- INT8 I/O는 입력과 출력이 INT8이지만 일부 연산은 FP32를 포함할 수 있어 “완전 정수 연산”으로 부르지 않습니다.',
      '- 각 측정 경로의 추론 라벨을 전체 평가 저장 예측과 대조했습니다. 1,000회 복원추출 벤치마크 자체는 새로운 정확도 평가가 아닙니다.','',
      '## 원시 기록과 재현','','- 측정 환경·의존성: `environment.json`, `requirements.txt`; 반복 지연: `*-default.csv`, `*-1.csv`; 케이스 로그·JSON 및 요약 파일은 이 폴더에 있습니다.',
      '- 전처리: `.venv\\Scripts\\python.exe scripts\\prepare_pamap2_preprocessing.py`; 평가: `.venv\\Scripts\\python.exe scripts\\evaluate_pamap2.py`',
      '- 변환·평가: `.venv\\Scripts\\python.exe scripts\\quantize_evaluate_pamap2.py`; 벤치마크: `.venv\\Scripts\\python.exe scripts\\benchmark_pamap2_windows.py`',
      '- 표와 그림: `.venv\\Scripts\\python.exe scripts\\summarize_pamap2_results.py`']
    (out/'pamap2-windows-results.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    colors=['#777777','#555555','#2878B5','#55A868','#C44E52','#8172B3']; y=np.arange(len(rows)); labels=[r['label'] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0];ax.barh(y,[r['accuracy_percent'] for r in rows],color=colors,alpha=.85,label='Accuracy');ax.scatter([r['macro_f1_percent'] for r in rows],y,color='black',marker='|',s=180,label='Macro-F1');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(96.4,97.8);ax.set_xlabel('Score (%)');ax.set_title('Test-set quality');ax.legend()
    ax=axes[0,1];ax.barh(y,[r['default_median_ms'] for r in rows],color=colors,alpha=.85,label='Median');ax.scatter([r['default_p95_ms'] for r in rows],y,color='black',marker='|',s=180,label='p95');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Latency (ms, log scale)');ax.set_title('Default-thread inference');ax.legend()
    ax=axes[1,0];ax.barh(y,[r['artifact_kib'] for r in rows],color=colors,alpha=.85);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Model size (KiB, log scale)');ax.set_title('Saved artifact size')
    ax=axes[1,1];ax.barh(y,[r['default_peak_working_set_mib'] for r in rows],color=colors,alpha=.85);ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Peak working set (MiB)');ax.set_title('Default-thread process memory')
    for a in axes.flat:a.grid(axis='x',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('PAMAP2 Windows CPU (batch 1, 1,000 timed inputs)',fontsize=14);fig.savefig(out/'pamap2-windows-results.png',dpi=180);plt.close(fig)
    print(f'Wrote {out}/pamap2-windows-results.csv, .md, .png')


if __name__=='__main__':main()
