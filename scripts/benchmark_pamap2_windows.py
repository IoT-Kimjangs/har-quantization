"""Measure the six PAMAP2 CPU inference paths on Windows."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = ('keras_predict', 'keras_direct', 'tflite_fp32', 'tflite_fp16', 'tflite_int8_dynamic', 'tflite_int8_full')
ENV_THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def worker(case, mode, output):
    import numpy as np
    import psutil
    prep_dir = ROOT / 'data/processed/pamap2-author-code'
    prep = json.loads((prep_dir / 'preprocessing-summary.json').read_text(encoding='utf-8'))
    quant_dir = ROOT / 'results/2026-11-10-windows-pamap2-quantization'
    quant = json.loads((quant_dir / 'metrics.json').read_text(encoding='utf-8'))
    model_config = json.loads((ROOT / 'configs/pamap2-model.json').read_text(encoding='utf-8'))
    model_path = ROOT / model_config['local_path']
    tflite_path = ROOT / f'models/pamap2_transformer_{case.removeprefix("tflite_")}.tflite'
    path = model_path if case.startswith('keras') else tflite_path
    expected_sha = model_config['sha256'] if case.startswith('keras') else quant['variants'][case.removeprefix('tflite_')]['sha256']
    if digest(path) != expected_sha:
        raise ValueError(f'{case}: model hash mismatch')
    config = json.loads((ROOT / 'configs/benchmark.json').read_text(encoding='utf-8'))
    inputs_all = np.load(prep_dir / 'X_test_s.npy', mmap_mode='r', allow_pickle=False)
    rng = np.random.RandomState(config['seed'])
    selected = rng.choice(len(inputs_all), config['measurement_iterations'], replace=len(inputs_all) < config['measurement_iterations'])
    inputs = np.ascontiguousarray(inputs_all[selected], dtype=np.float32)
    del inputs_all
    samples = [inputs[i:i+1] for i in range(len(inputs))]
    window_ids = np.load(prep_dir / 'test_window_ids.npy', allow_pickle=False)[selected]
    if case.startswith('keras'):
        with (ROOT / 'results/2026-11-10-windows-pamap2-fp32/predictions.csv').open(newline='', encoding='utf-8') as stream:
            full = list(csv.DictReader(stream))
        prior_labels = np.array([int(row['keras_label']) for row in full])
        os.environ['KERAS_BACKEND'] = 'tensorflow'; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ.setdefault('KERAS_HOME', str(Path(sys.prefix)/'.keras'))
        os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix)/'.matplotlib'))
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
        import keras, tensorflow as tf
        sys.path.insert(0, str(ROOT/'src'))
        from har_quantization.layers import AddPositionEmbedding
        tf.config.set_visible_devices([], 'GPU')
        if mode == '1':
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        model = keras.models.load_model(model_path, custom_objects={'AddPositionEmbedding':AddPositionEmbedding}, compile=False, safe_mode=True)
        invoke = (lambda s: model.predict(s, batch_size=1, verbose=0)) if case == 'keras_predict' else (lambda s: model(s, training=False).numpy())
        runtime = {'tensorflow_intra_op':tf.config.threading.get_intra_op_parallelism_threads(), 'tensorflow_inter_op':tf.config.threading.get_inter_op_parallelism_threads()}
    else:
        with (quant_dir/'predictions.csv').open(newline='',encoding='utf-8') as stream:
            full=list(csv.DictReader(stream))
        prior_labels=np.array([int(row[case.removeprefix('tflite_')]) for row in full])
        from ai_edge_litert.interpreter import Interpreter
        interpreter=Interpreter(model_path=str(path), **({'num_threads':1} if mode=='1' else {})); interpreter.allocate_tensors()
        inp,out=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
        def invoke(sample):
            if case == 'tflite_int8_full':
                scale,zero=inp['quantization']; value=np.clip(np.rint(sample/scale+zero),-128,127).astype(np.int8)
            else: value=sample
            interpreter.set_tensor(inp['index'],value); interpreter.invoke(); result=interpreter.get_tensor(out['index'])
            if case == 'tflite_int8_full':
                scale,zero=out['quantization']; result=(result.astype(np.float32)-zero)*scale
            return result
        runtime={'interpreter_num_threads_requested':1 if mode=='1' else None}
    warmup=config['warmup_iterations']; print(f'Warmup {warmup}; {len(samples)} PAMAP2 inputs: {case}/{mode}',flush=True)
    for i in range(warmup): invoke(samples[i % len(samples)])
    rows=[]; disagreements=0
    for i,sample in enumerate(samples):
        start=time.perf_counter_ns(); result=invoke(sample); elapsed=time.perf_counter_ns()-start
        if result.shape != (1,18) or not np.isfinite(result).all(): raise ValueError(f'{case}: invalid output {result.shape}')
        pred=int(result.argmax(axis=1)[0]); disagreements += int(pred != prior_labels[selected[i]])
        rows.append((i,int(selected[i]),int(window_ids[i]),elapsed,pred))
    raw=output/f'{case}-{mode}.csv'
    with raw.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream,lineterminator='\n'); writer.writerow(['iteration','test_index','window_id','latency_ns','predicted_label']); writer.writerows(rows)
    durations=np.asarray([row[3] for row in rows],dtype=np.float64)/1e6
    mem=psutil.Process(os.getpid()).memory_info()
    record={'variant':case,'thread_mode':mode,'iterations':len(rows),'warmup':warmup,'batch_size':1,
            'test_windows':len(prior_labels),'sampling_with_replacement':len(prior_labels)<len(rows),
            'median_ms':float(np.median(durations)),'p95_ms':float(np.percentile(durations,95)),'mean_ms':float(durations.mean()),
            'runtime_threads':runtime,'prediction_disagreements_with_prior_full_evaluation':disagreements,
            'model_bytes':path.stat().st_size,'model_sha256':digest(path),'raw_csv':raw.name,'raw_csv_sha256':digest(raw),
            'peak_working_set_bytes':getattr(mem,'peak_wset',mem.rss),'final_rss_bytes':mem.rss}
    save_json(output/f'{case}-{mode}.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=Path('results/2026-11-10-windows-pamap2-benchmark-final'))
    parser.add_argument('--worker-case',choices=CASES); parser.add_argument('--worker-mode',choices=['1','default']); args=parser.parse_args()
    if platform.system()!='Windows' or sys.prefix==sys.base_prefix: raise ValueError('Run in the Windows project virtual environment')
    output=args.output if args.output.is_absolute() else ROOT/args.output
    if args.worker_case: worker(args.worker_case,args.worker_mode,output); return
    if output.exists() and any(output.iterdir()): raise ValueError('Refusing to overwrite existing benchmark data; use a fresh output directory')
    import numpy as np, psutil
    config=json.loads((ROOT/'configs/benchmark.json').read_text(encoding='utf-8'))
    if config['measurement_iterations']!=1000 or config['warmup_iterations']!=50 or config['batch_size']!=1: raise ValueError('Benchmark protocol mismatch')
    output.mkdir(parents=True,exist_ok=True)
    plan=[(case,str(mode)) for mode in config['threads'] for case in config['variants']]; np.random.RandomState(config['seed']).shuffle(plan)
    power=subprocess.run(['powercfg','/getactivescheme'],capture_output=True)
    env_record={'platform':platform.platform(),'architecture':platform.machine(),'python':platform.python_version(),
                'cpu':platform.processor() or 'unknown','ram_gb':round(psutil.virtual_memory().total/1024**3,2),'logical_cpu_count':os.cpu_count(),
                'packages':{p:importlib.metadata.version(p) for p in ('tensorflow','keras','ai-edge-litert','numpy','psutil')},
                'power_plan':power.stdout.decode(errors='replace').strip(),'seed':config['seed'],'measurement_iterations':1000,
                'warmup_iterations':50,'test_windows':494,'sampling':'seeded test-window sample; replacement when population < 1000',
                'note':'CPU run; background applications, CPU frequency and thermal state were not controlled.'}
    save_json(output/'environment.json',env_record)
    (output/'requirements.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    summaries=[]; started=time.time()
    for case,mode in plan:
        env=os.environ.copy()
        for key in ENV_THREADS: env.pop(key,None)
        if mode=='1': env.update({key:'1' for key in ENV_THREADS})
        env.update(KERAS_BACKEND='tensorflow',CUDA_VISIBLE_DEVICES='-1',KERAS_HOME=str(Path(sys.prefix)/'.keras'),MPLCONFIGDIR=str(Path(sys.prefix)/'.matplotlib'),TF_CPP_MIN_LOG_LEVEL='2')
        print(f'Start {len(summaries)+1}/12: {case}, threads={mode}',flush=True)
        log_path=output/f'{case}-{mode}.log'; peak=threads=0
        with log_path.open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output',str(output),'--worker-case',case,'--worker-mode',mode],env=env,stdout=log,stderr=subprocess.STDOUT)
            observed=psutil.Process(proc.pid)
            while proc.poll() is None:
                try: peak=max(peak,observed.memory_info().rss); threads=max(threads,observed.num_threads())
                except psutil.NoSuchProcess: pass
                time.sleep(.01)
            if proc.returncode: raise RuntimeError(f'{case}/{mode} failed; inspect {log_path.name}')
        record=json.loads((output/f'{case}-{mode}.json').read_text(encoding='utf-8'))
        record.update(parent_sampled_peak_rss_bytes=peak,sampled_peak_os_threads=threads)
        record['runtime_threads']['xnnpack_log_confirmed']='Created TensorFlow Lite XNNPACK delegate for CPU' in log_path.read_text(encoding='utf-8',errors='replace') if case.startswith('tflite') else None
        save_json(output/f'{case}-{mode}.json',record); summaries.append(record)
        save_json(output/'summary.json',{'cases':summaries,'elapsed_seconds':time.time()-started})
        print(f"Done: median {record['median_ms']:.6f} ms, p95 {record['p95_ms']:.6f} ms, peak WS {record['peak_working_set_bytes']/1024**2:.1f} MiB",flush=True)
    with (output/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['variant','thread_mode','iterations','sampling_with_replacement','median_ms','p95_ms','mean_ms','peak_working_set_bytes','final_rss_bytes','prediction_disagreements_with_prior_full_evaluation','model_bytes']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n',extrasaction='ignore'); writer.writeheader(); writer.writerows(summaries)
    save_json(output/'summary.json',{'cases':summaries,'elapsed_seconds':time.time()-started,
        'timing_scope':'Inference call plus tensor input/output. Model loading and file I/O excluded; INT8 input quantize and output dequantize included.',
        'memory_scope':'Worker-reported Windows peak working set includes imports, model and selected inputs.',
        'accuracy_scope':'Each sampled output compared with full-test-set saved predictions; sample size 1000 with replacement because test set has 494 windows.'})
    print(f'Completed 12 conditions in {time.time()-started:.1f}s: {output}')


if __name__=='__main__': main()
