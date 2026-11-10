"""Load both pinned HAR models and record their parameter counts/signatures."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('KERAS_BACKEND','tensorflow')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')

import keras

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from har_quantization.layers import AddPositionEmbedding


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    records=[]
    for dataset in ('wisdm','pamap2'):
        path=ROOT/f'configs/{dataset}-model.json'; cfg=json.loads(path.read_text(encoding='utf-8'))
        model_path=ROOT/cfg['local_path']
        if sha(model_path)!=cfg['sha256']:raise ValueError(f'{dataset} model hash mismatch')
        model=keras.models.load_model(model_path,custom_objects={'AddPositionEmbedding':AddPositionEmbedding},compile=False,safe_mode=True)
        cfg['parameter_count']=int(model.count_params())
        cfg['input_shape']=[None,*[int(x) if x is not None else None for x in model.input_shape[1:]]]
        cfg['output_shape']=[None,int(model.output_shape[-1])]
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        records.append({'dataset':dataset,'parameter_count':cfg['parameter_count'],'input_shape':cfg['input_shape'],'output_shape':cfg['output_shape']})
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
