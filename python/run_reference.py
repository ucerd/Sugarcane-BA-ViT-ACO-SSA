"""Run new reference experiments. Never presents output as historical paper results."""
import argparse, json, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from reference import BAViT, CLASSES, read_manifest, hybrid_select, classifiers, metrics

class Images(Dataset):
    def __init__(self,t,root,training=False,size=384):
        self.t=t.reset_index(drop=True); self.root=Path(root)
        aug=[transforms.RandomHorizontalFlip(),transforms.RandomRotation(15),transforms.ColorJitter(.1,.1,.1,.02)] if training else []
        self.transform=transforms.Compose([transforms.Resize((size,size)),*aug,transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.t)
    def __getitem__(self,i):
        row=self.t.iloc[i]
        with Image.open(self.root/row.relative_path) as im: x=self.transform(im.convert('RGB'))
        return x,CLASSES.index(row.variety)

def loader(t,root,cfg,train=False,size=384):
    return DataLoader(Images(t,root,train,size),batch_size=cfg['batch_size'],shuffle=train,num_workers=0)

def fit(model,dl,cfg,device):
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']); criterion=torch.nn.CrossEntropyLoss()
    for _ in range(cfg['epochs']):
        model.train()
        for x,y in dl:
            opt.zero_grad(); loss=criterion(model(x.to(device)),y.to(device)); loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def infer(model,dl,device,features=False):
    return np.concatenate([(model.features(x.to(device)) if features else model(x.to(device))).cpu().numpy() for x,_ in dl])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True); p.add_argument('--root',required=True); p.add_argument('--output',required=True); p.add_argument('--config',default='configs/reference.json'); p.add_argument('--new-folds',action='store_true'); p.add_argument('--alexnet',action='store_true'); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text()); t=read_manifest(a.manifest,a.root,a.new_folds,cfg['seed']); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    t.to_csv(out/'effective_manifest.csv',index=False); (out/'run.json').write_text(json.dumps({'status':'NEW REFERENCE EXPERIMENT','fold_provenance':'generated' if a.new_folds else 'user supplied, historical identity unverified','config':cfg,'python':platform.python_version(),'torch':torch.__version__},indent=2))
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if cfg['device']=='auto' else cfg['device']); predictions=[]
    torch.use_deterministic_algorithms(True,warn_only=True)
    for f in range(1,6):
        seed=cfg['seed']+f; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        train=t[t.fold!=f]; test=t[t.fold==f]; y=train.variety.map(CLASSES.index).to_numpy(); yt=test.variety.map(CLASSES.index).to_numpy(); fd=out/f'fold_{f}'; fd.mkdir()
        model=fit(BAViT(cfg['width'],cfg['depth']),loader(train,a.root,cfg,True),cfg,device)
        X=infer(model,loader(train,a.root,cfg),device,True); Z=infer(model,loader(test,a.root,cfg),device,True)
        torch.save(model.state_dict(),fd/'reference_bavit.pt'); np.savez_compressed(fd/'features.npz',X=X,Z=Z,y=y,yt=yt)
        for n,feat,rows in [('train',X,train),('test',Z,test)]:
            df=pd.DataFrame(feat,columns=[f'f{i+1}' for i in range(feat.shape[1])]); df.insert(0,'image_id',rows.image_id.to_numpy()); df.to_csv(fd/f'{n}_features.csv',index=False)
        for k in cfg['feature_counts']:
            idx,h=hybrid_select(X,y,train.source_image_id.to_numpy(),k,seed,**{n:cfg[n] for n in ['population','iterations','evaporation','w1','w2']}); np.save(fd/f'mask_{k}.npy',idx); np.savetxt(fd/f'fitness_{k}.csv',h,delimiter=',')
            for name,clf in classifiers(k).items():
                if name=='CrKNN' and len(y)<100: continue
                clf.fit(X[:,idx],y); pred=clf.predict(Z[:,idx]); predictions.append(pd.DataFrame({'image_id':test.image_id,'fold':f,'model':name,'features':k,'truth':yt,'prediction':pred}))
        if a.alexnet:
            alex=models.alexnet(weights=models.AlexNet_Weights.DEFAULT); alex.classifier[6]=torch.nn.Linear(4096,5)
            alex=fit(alex,loader(train,a.root,cfg,True,224),cfg,device); scores=infer(alex,loader(test,a.root,cfg,False,224),device); torch.save(alex.state_dict(),fd/'reference_alexnet.pt'); np.save(fd/'alexnet_logits.npy',scores)
            predictions.append(pd.DataFrame({'image_id':test.image_id,'fold':f,'model':'AlexNet','features':0,'truth':yt,'prediction':scores.argmax(1)}))
        pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    df=pd.concat(predictions); rows=[]
    for (name,k),g in df.groupby(['model','features']):
        for fold,h in [('pooled',g),*list(g.groupby('fold'))]: rows.append({'model':name,'features':int(k),'fold':str(fold),**metrics(h.truth,h.prediction)})
    (out/'metrics.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
