"""Apply an AUTHOR-SUPPLIED TorchScript ESRGAN-compatible RGB model; not historical settings."""
import argparse, hashlib, json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--scale',required=True,type=int)
a=p.parse_args(); src=Path(a.input); dst=Path(a.output)
if dst.exists(): raise FileExistsError(dst)
net=torch.jit.load(a.model,map_location='cpu').eval()
with Image.open(src) as im: x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None]
with torch.no_grad(): y=net(x)
assert y.shape==(1,3,x.shape[-2]*a.scale,x.shape[-1]*a.scale), 'Unexpected model output shape'
dst.parent.mkdir(parents=True,exist_ok=True); Image.fromarray((y[0].permute(1,2,0).clamp(0,1).numpy()*255).round().astype('uint8')).save(dst)
dst.with_suffix(dst.suffix+'.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),'scale':a.scale,'status':'new enhancement; historical recipe unavailable'},indent=2))
