"""Compare prepared PAMAP2 arrays against the pinned author's two preprocessing functions."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prepare_pamap2 import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/pamap2.json').read_text())
    if sha256(args.notebook)!=config['notebook_sha256']:
        raise ValueError('기준 노트북 해시가 다릅니다.')
    notebook=json.loads(args.notebook.read_text())
    functions=[]
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in ('load_pamap2','make_windows'):
                functions.append(node)
    if len(functions)!=2:raise ValueError('원본 로딩·윈도우 함수 두 개를 찾지 못했습니다.')
    namespace={'np':np,'pd':pd,'FEATURE_IDXS':config['feature_columns_zero_based'],
               'FEATURE_NAMES':config['feature_names'],'WIN':200,'STEP':100}
    # Execute only these reviewed, hash-pinned function definitions, never the notebook cells/training.
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(args.notebook),'exec'),namespace)
    folder=ROOT/config['processed_directory']
    prep=json.loads((folder/'preprocessing-summary.json').read_text())
    clean=np.memmap(folder/'clean-selected-columns.bin',dtype=np.float64,mode='r',shape=(prep['clean_rows'],20))
    for item in prep['raw_files']:
        path=ROOT/config['raw_directory']/item['path']
        frame=namespace['load_pamap2']([str(path)])
        frame=frame[frame.activity_id!=0]
        frame=frame.replace([np.inf,-np.inf],np.nan).dropna()
        np.testing.assert_array_equal(frame.to_numpy(),clean[item['clean_start']:item['clean_end']])
        print('원본 로더와 일치:',item['path'],flush=True)
    values=clean[:,1:].astype(np.float32)
    ids=config['activity_ids'];mapping={a:i for i,a in enumerate(ids)}
    labels=np.array([mapping[int(a)] for a in clean[:,0]],dtype=np.int32)
    X,y=namespace['make_windows'](values,labels)
    np.testing.assert_array_equal(X,np.load(folder/'X_windows.npy',mmap_mode='r'))
    train,test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler=StandardScaler()
    train_scaled=scaler.fit_transform(train.reshape(-1,19)).reshape(train.shape)
    test_scaled=scaler.transform(test.reshape(-1,19)).reshape(test.shape)
    for name,array in [('X_train_s.npy',train_scaled),('X_test_s.npy',test_scaled),('y_train.npy',y_train),('y_test.npy',y_test)]:
        np.testing.assert_array_equal(array,np.load(folder/name,mmap_mode='r'))
    report={'notebook_sha256':config['notebook_sha256'],'source_commit':config['source_commit'],
            'raw_files_checked':len(prep['raw_files']),'exact_cleaned_rows_match':True,
            'exact_windows_match':True,'exact_train_test_arrays_match':True,
            'test_windows':len(test),'notebook_training_executed':False}
    (ROOT/'results/pamap2-preparation/reference-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
