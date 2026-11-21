import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import pandas as pd
import torch
import pytest
from reference import BAViT, CLASSES, read_manifest, hybrid_select, metrics

def test_model_shape():
    torch.set_num_threads(1)
    m=BAViT(32,1).eval()
    with torch.no_grad(): assert m.features(torch.zeros(1,3,384,384)).shape==(1,1024)

def test_groups(tmp_path):
    records=[]
    for c in CLASSES:
        for j in range(5):
            key=f'{c}_{j}'
            for suffix in ['', '_d']:
                i=key+suffix; path=i+'.jpg'; (tmp_path/path).touch(); records.append([i,path,c,key,int(not suffix)])
    t=pd.DataFrame(records,columns=['image_id','relative_path','variety','source_image_id','is_original']); p=tmp_path/'m.csv'; t.to_csv(p,index=False)
    t=read_manifest(p,tmp_path,True); assert t.groupby('source_image_id').fold.nunique().max()==1
    t.loc[1,'fold']=t.loc[0,'fold']%5+1; t.to_csv(p,index=False)
    with pytest.raises(ValueError,match='leakage'): read_manifest(p,tmp_path)

def test_selector():
    rng=np.random.default_rng(5); y=np.repeat(np.arange(5),12); X=rng.normal(size=(len(y),20)); X[:,0]+=y; groups=np.arange(len(y)); idx,h=hybrid_select(X,y,groups,4,iterations=4,population=3); assert len(set(idx))==4 and np.all(np.diff(h)<=0)

def test_metrics():
    y=np.repeat(np.arange(5),2); m=metrics(y,y); assert m['macro_f1']==1 and m['accuracy']==1
