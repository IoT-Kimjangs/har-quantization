"""Evaluate PAMAP2 Keras and four LiteRT variants on the same full test set."""
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

from prepare_pamap2 import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('프로젝트 가상환경의 Python으로 실행해주세요.')
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads((ROOT / 'configs/pamap2.json').read_text())
    prepared = ROOT / config['processed_directory']
    preparation = json.loads((prepared / 'preprocessing-summary.json').read_text())
    source = ROOT / config['model_path']
    paths = {name: ROOT / f'models/pamap2_transformer_{name}.tflite'
             for name in ('fp32', 'fp16', 'int8_dynamic', 'int8_full')}
    if any(p.exists() for p in paths.values()) or (output.exists() and any(output.iterdir())):
        raise ValueError('기존 모델·결과를 덮어쓰지 않습니다.')
    if sha256(source) != config['model_sha256']:
        raise ValueError('원본 모델 해시가 다릅니다.')
    for name, expected in preparation['artifact_sha256'].items():
        if sha256(prepared / name) != expected:
            raise ValueError(f'전처리 산출물 해시가 다릅니다: {name}')
    os.environ.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1')
    os.environ.setdefault('KERAS_HOME', str(Path(sys.prefix) / '.keras'))
    os.environ.setdefault('MPLCONFIGDIR', str(Path(sys.prefix) / '.matplotlib'))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    import keras
    from ai_edge_litert.interpreter import Interpreter
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    tf.config.set_visible_devices([], 'GPU')
    inputs = np.load(prepared / 'X_test_s.npy', allow_pickle=False)
    truth = np.load(prepared / 'y_test.npy', allow_pickle=False)
    train = np.load(prepared / 'X_train_s.npy', mmap_mode='r', allow_pickle=False)
    test_ids = np.load(prepared / 'test_window_ids.npy', allow_pickle=False)
    train_ids = np.load(prepared / 'train_window_ids.npy', allow_pickle=False)
    classes = len(config['activity_ids'])
    if inputs.shape[1:] != (200,19) or list(inputs.shape) != preparation['test_shape']:
        raise ValueError('평가 입력 형태가 다릅니다.')
    if truth.shape != (len(inputs),) or test_ids.shape != truth.shape:
        raise ValueError('정답·구간 번호 수가 다릅니다.')
    if not np.isfinite(inputs).all() or not np.isin(truth, np.arange(classes)).all():
        raise ValueError('잘못된 입력 또는 정답입니다.')
    selected = np.random.RandomState(config['seed']).choice(len(train), min(300,len(train)), replace=False)
    if np.intersect1d(train_ids[selected], test_ids).size:
        raise ValueError('보정에 평가 구간 번호가 포함됐습니다.')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'calibration.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['train_index','window_id'])
        writer.writerows((int(i),int(train_ids[i])) for i in selected)

    def representative():
        for i in selected:
            yield [np.asarray(train[i:i+1],dtype=np.float32)]

    def metrics(scores):
        if scores.shape != (len(truth), classes) or not np.isfinite(scores).all():
            raise ValueError('출력 형태·유한성 검사 실패')
        predicted=scores.argmax(axis=1)
        matrix=confusion_matrix(truth,predicted,labels=np.arange(classes))
        accuracy=float(accuracy_score(truth,predicted)*100)
        f1=float(f1_score(truth,predicted,labels=np.arange(classes),average='macro',zero_division=0)*100)
        denominator=matrix.sum(0)+matrix.sum(1)
        manual=np.divide(2*matrix.diagonal(),denominator,out=np.zeros(classes),where=denominator!=0).mean()*100
        np.testing.assert_allclose(f1,manual,atol=1e-12)
        np.testing.assert_allclose(accuracy,np.trace(matrix)/len(truth)*100,atol=1e-12)
        return {'accuracy_percent':accuracy,'macro_f1_percent':f1,'correct_windows':int(np.trace(matrix)),
                'evaluated_windows':len(truth),'confusion_matrix':matrix.tolist()},predicted

    model=keras.models.load_model(source,compile=False,safe_mode=True)
    if model.input_shape != (None,200,19) or model.output_shape != (None,18):
        raise ValueError('저자 모델 입출력 형태가 다릅니다.')
    print(f'Keras 전체 평가: {len(inputs)}개',flush=True)
    keras_scores=np.concatenate([model(inputs[i:i+128],training=False).numpy() for i in range(0,len(inputs),128)])
    keras_metrics,keras_labels=metrics(keras_scores)
    print(json.dumps(keras_metrics | {'confusion_matrix':'saved in report'}),flush=True)
    report={'dataset':'PAMAP2','platform':platform.platform(),'architecture':platform.machine(),'python':platform.python_version(),
            'packages':{p:importlib.metadata.version(p) for p in ('tensorflow','keras','ai-edge-litert','numpy','scikit-learn')},
            'source_keras_sha256':sha256(source),'source_keras_bytes':source.stat().st_size,
            'preprocessing_summary_sha256':sha256(prepared/'preprocessing-summary.json'),
            'preprocessing_artifact_sha256':preparation['artifact_sha256'],
            'activity_ids':config['activity_ids'],'activity_names':config['activity_names'],
            'keras_reference':keras_metrics,'variants':{},'interpreter_threads':1,
            'calibration':{'windows':len(selected),'seed':config['seed'],'source':'X_train_s.npy','test_window_id_overlap':0,
                           'indices_sha256':sha256(output/'calibration.csv')},
            'paper_reference':{'accuracy_percent':98.480,'macro_f1_percent':97.978,'pdf_page':11,'table':3,
                               'doi':'10.1038/s41598-026-62744-4'},
            'keras_difference_from_paper_pp':{'accuracy':keras_metrics['accuracy_percent']-98.480,
                                             'macro_f1':keras_metrics['macro_f1_percent']-97.978},
            'training_executed':False,'latency_measured':False,'matches_saved_model_training_statistics':'unverified'}
    predictions={'keras_label':keras_labels}
    with tempfile.TemporaryDirectory(prefix='pamap2-export-',dir=ROOT/'models') as tmp:
        export=Path(tmp)/'saved_model';model.export(str(export),format='tf_saved_model',verbose=False)
        for name,path in paths.items():
            print(f'변환·평가: {name}',flush=True)
            try:
                converter=tf.lite.TFLiteConverter.from_saved_model(str(export))
                converter.optimizations=[] if name=='fp32' else [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
                if name=='fp16':converter.target_spec.supported_types=[tf.float16]
                if name=='int8_full':
                    converter.representative_dataset=representative
                    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
                path.write_bytes(converter.convert())
                inter=Interpreter(model_path=str(path),num_threads=1);inter.allocate_tensors()
                inp,out=inter.get_input_details()[0],inter.get_output_details()[0]
                expected_dtype=np.int8 if name=='int8_full' else np.float32
                if inp['dtype']!=expected_dtype or out['dtype']!=expected_dtype:
                    raise ValueError('예상 입출력 dtype과 다릅니다.')
                if list(inp['shape'])!=[1,200,19] or list(out['shape'])!=[1,classes]:
                    raise ValueError('TFLite 입출력 형태가 다릅니다.')
                in_scale,in_zero=inp['quantization'];out_scale,out_zero=out['quantization']
                if name=='int8_full' and (in_scale<=0 or out_scale<=0):raise ValueError('INT8 scale 오류')
                scores=np.empty((len(inputs),classes),dtype=np.float32);clipped=0
                for i in range(len(inputs)):
                    sample=inputs[i:i+1]
                    if name=='int8_full':
                        q=np.rint(sample/in_scale+in_zero);clipped+=int(np.count_nonzero((q < -128)|(q >127)))
                        sample=np.clip(q,-128,127).astype(np.int8)
                    inter.set_tensor(inp['index'],sample);inter.invoke();value=inter.get_tensor(out['index'])[0]
                    if name=='int8_full':value=(value.astype(np.float32)-out_zero)*out_scale
                    scores[i]=value
                if (scores<0).any() or (scores>1).any():raise ValueError('확률 범위 오류')
                np.testing.assert_allclose(scores.sum(1),1,atol=(classes*out_scale+1e-5 if name=='int8_full' else 1e-5))
                result,predicted=metrics(scores)
                details=inter.get_tensor_details();dtypes=Counter(np.dtype(t['dtype']).name for t in details)
                float_ids={t['index'] for t in details if np.issubdtype(t['dtype'],np.floating)}
                float_ops=[]
                if name=='int8_full':
                    for op in inter._get_ops_details():
                        if op['op_name']!='DELEGATE' and any(int(i) in float_ids for i in [*op['inputs'],*op['outputs']]):
                            float_ops.append({'operator':op['op_name'],'inputs':[int(i) for i in op['inputs']],
                                              'outputs':[int(i) for i in op['outputs']]})
                if name=='fp16' and not dtypes['float16']:raise ValueError('FP16 텐서 없음')
                if name.startswith('int8') and not dtypes['int8']:raise ValueError('INT8 텐서 없음')
                result.update(status='passed',bytes=path.stat().st_size,kib=path.stat().st_size/1024,sha256=sha256(path),
                    prediction_agreement_with_keras_percent=float(np.mean(predicted==keras_labels)*100),
                    max_absolute_error_from_keras=float(np.max(np.abs(scores.astype(np.float64)-keras_scores))),
                    tensor_dtypes=dict(dtypes),integer_only_achieved=(not float_ids if name=='int8_full' else None),
                    floating_operators=float_ops,input_dtype=str(np.dtype(inp['dtype'])),output_dtype=str(np.dtype(out['dtype'])),
                    input_quantization=[float(in_scale),int(in_zero)],output_quantization=[float(out_scale),int(out_zero)],
                    input_clipped_values=clipped,input_values=int(inputs.size),select_tf_ops_enabled=False)
                if name=='fp32':
                    report['fp32_baseline']=result;report['fp32_model_sha256']=result['sha256']
                else:report['variants'][name]=result
                predictions['fp32_label' if name=='fp32' else name]=predicted
                print(json.dumps({k:result[k] for k in ('kib','accuracy_percent','macro_f1_percent','integer_only_achieved')}),flush=True)
            except Exception as error:
                report['variants'][name]={'status':'failed','error_type':type(error).__name__,'error':str(error)}
                print(f'{name} 실패: {error}',flush=True)
            (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'predictions.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['test_index','window_id','true_label',*predictions])
        writer.writerows((i,int(test_ids[i]),int(truth[i]),*(int(v[i]) for v in predictions.values())) for i in range(len(truth)))
    report['predictions_sha256']=sha256(output/'predictions.csv')
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if 'fp32_baseline' in report and all(r['status']=='passed' for r in report['variants'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
