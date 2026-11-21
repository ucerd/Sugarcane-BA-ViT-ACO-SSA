"""New reference implementation; not the missing historical experiment."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torch import nn

CLASSES = ['CP77-400','CPF-237','CPF-250','CPF-251','CPF-253']
COLS = ['image_id','relative_path','variety','source_image_id','is_original']

def read_manifest(path, root, new_folds=False, seed=42):
    t = pd.read_csv(path, dtype={c:str for c in COLS})
    if not set(COLS).issubset(t) or t.empty or t[COLS].isna().any().any():
        raise ValueError('Missing manifest fields or rows')
    if t.image_id.duplicated().any() or t.relative_path.duplicated().any():
        raise ValueError('Duplicate IDs or paths')
    if set(t.variety) != set(CLASSES): raise ValueError('Expected five manuscript classes')
    v = t.is_original.str.lower().map({'1':True,'0':False,'true':True,'false':False})
    if v.isna().any(): raise ValueError('Invalid is_original')
    t['is_original'] = v
    root = Path(root).resolve()
    for p in t.relative_path:
        pp = (root/p).resolve()
        if not pp.is_relative_to(root) or not pp.is_file(): raise ValueError(f'Invalid image path: {p}')
    for g, rows in t.groupby('source_image_id'):
        if rows.variety.nunique()!=1 or rows.is_original.sum()!=1 or rows.loc[rows.is_original,'image_id'].iloc[0]!=g:
            raise ValueError(f'Invalid parent group {g}')
    if new_folds:
        groups=t.loc[t.is_original].sort_values('image_id').reset_index(drop=True)
        f=np.zeros(len(groups),dtype=int)
        for i,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(groups,groups.variety),1): f[te]=i
        t['fold']=t.source_image_id.map(dict(zip(groups.image_id,f)))
    if 'fold' not in t or not t.fold.isin(range(1,6)).all(): raise ValueError('Provide historical fold IDs or explicitly generate new folds')
    if (t.groupby('source_image_id').fold.nunique()!=1).any(): raise ValueError('Group leakage')
    for f in range(1,6):
        if set(t.loc[t.fold==f,'variety'])!=set(CLASSES): raise ValueError('Fold lacks classes')
    return t

class BAViT(nn.Module):
    """Reference ViT with manuscript patch/head/output constraints; new depth and width."""
    def __init__(self, width=192, depth=4, num_classes=5):
        super().__init__()
        self.patch=nn.Conv2d(3,width,16,16)
        self.pos=nn.Parameter(torch.zeros(1,576,width))
        nn.init.trunc_normal_(self.pos,std=.02)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,width*4,batch_first=True,norm_first=True),depth,enable_nested_tensor=False)
        self.attention=nn.MultiheadAttention(width,8,batch_first=True)
        self.norm=nn.LayerNorm(width)
        self.projection=nn.Linear(width,1024)
        self.head=nn.Linear(1024,num_classes)
    def features(self,x):
        x=self.patch(x).flatten(2).transpose(1,2)+self.pos
        x=self.encoder(x)
        a,_=self.attention(x,x,x,need_weights=False)
        return self.projection(self.norm(x+a).mean(1))
    def forward(self,x): return self.head(self.features(x))

def classifiers(k):
    # New reference hyperparameters; not MATLAB Classification Learner presets.
    return {
      'LSVM':make_pipeline(StandardScaler(),SVC(kernel='linear')),
      'QSVM':make_pipeline(StandardScaler(),SVC(kernel='poly',degree=2,coef0=1)),
      'CuSVM':make_pipeline(StandardScaler(),SVC(kernel='poly',degree=3,coef0=1)),
      'FGSVM':make_pipeline(StandardScaler(),SVC(gamma=8/k)),
      'MGSVM':make_pipeline(StandardScaler(),SVC(gamma=1/k)),
      'CGSVM':make_pipeline(StandardScaler(),SVC(gamma=.125/k)),
      'FKNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=1)),
      'CKNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=10,metric='cosine')),
      'CrKNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=100))}

def hybrid_select(X,y,groups,k,seed=42,population=10,iterations=100,evaporation=.2,w1=.99,w2=.01):
    """New fixed-cardinality ACO exploration then continuous SSA refinement.
    Fitness computed on training-side grouped inner holdout only. Fixed k makes size penalty constant.
    """
    X=np.asarray(X); y=np.asarray(y); groups=np.asarray(groups); d=X.shape[1]
    if not 1<=k<=d or iterations<2 or population<2: raise ValueError('Invalid search settings')
    rng=np.random.default_rng(seed)
    a,b=next(StratifiedGroupKFold(3,shuffle=True,random_state=seed).split(X,y,groups))
    assert not set(groups[a])&set(groups[b])
    if set(y[a])!=set(y) or set(y[b])!=set(y): raise ValueError('Inner split lacks classes')
    best=None; loss=np.inf; history=[]; pher=np.ones(d); cache={}
    def score(idx):
        key=tuple(sorted(idx))
        if key not in cache:
            model=classifiers(k)['CuSVM']; model.fit(X[a][:,key],y[a]); pred=model.predict(X[b][:,key]); cache[key]=w1*(1-accuracy_score(y[b],pred))+w2*k/d
        return cache[key]
    def update(idx):
        nonlocal best,loss
        val=score(idx)
        if val<loss: best=np.array(sorted(idx)); loss=val
        return val
    for _ in range(iterations//2):
        pher*=1-evaporation
        for _ in range(population): update(rng.choice(d,k,replace=False,p=pher/pher.sum()))
        pher[best]+=1/(1+loss); history.append(loss)
    pos=rng.random((population,d)); pos[0,best]=1
    for step in range(iterations-iterations//2):
        food=np.zeros(d); food[best]=1
        c1=2*np.exp(-(4*(step+1)/(iterations-iterations//2))**2)
        pos[0]=np.clip(food+c1*rng.uniform(-1,1,d),0,1)
        for j in range(1,population): pos[j]=(pos[j]+pos[j-1])/2
        for row in pos: update(np.argpartition(row,-k)[-k:])
        history.append(loss)
    return best,np.array(history)

def metrics(y,pred):
    c=confusion_matrix(y,pred,labels=range(5)); tp=c.diagonal(); fp=c.sum(0)-tp; fn=c.sum(1)-tp; tn=c.sum()-tp-fp-fn
    def divide(a,b): return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    p=divide(tp,tp+fp); r=divide(tp,tp+fn); s=divide(tn,tn+fp); f=divide(2*p*r,p+r); g=np.sqrt(r*s)
    return {'accuracy':float(tp.sum()/c.sum()),'macro_precision':float(p.mean()),'macro_recall':float(r.mean()),'macro_f1':float(f.mean()),'macro_specificity':float(s.mean()),'macro_gmean':float(g.mean()),'confusion':c.tolist()}
