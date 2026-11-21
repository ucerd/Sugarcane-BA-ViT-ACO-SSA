"""Download official parent dataset and verify its user-provided published MD5."""
import argparse, hashlib, json, urllib.request, zipfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__); p.add_argument('--dest',default='downloads'); p.add_argument('--download',action='store_true'); p.add_argument('--extract',action='store_true'); a=p.parse_args()
d=Path(a.dest); d.mkdir(parents=True,exist_ok=True)
url='https://zenodo.org/api/records/18619387'
with urllib.request.urlopen(url,timeout=60) as response: meta=json.load(response)
(d/'zenodo_18619387.json').write_text(json.dumps(meta,indent=2))
file=next(f for f in meta['files'] if f['key']=='Sugarcane_Varieties_Dataset.zip')
expected='2ea3cdc0ba74710e154f6353f5700519'
if file['checksum']!='md5:'+expected: raise ValueError('Published record checksum differs from supplied record')
z=d/file['key']
if a.download:
    if not z.exists():
        tmp=z.with_suffix('.part'); urllib.request.urlretrieve(file['links']['self'],tmp); tmp.rename(z)
if a.download or a.extract:
    with z.open('rb') as f: digest=hashlib.file_digest(f,'md5').hexdigest()
    if digest!=expected: raise ValueError('Archive checksum mismatch')
if a.extract:
    root=(d/'parent').resolve(); root.mkdir(exist_ok=False)
    with zipfile.ZipFile(z) as zz:
        for item in zz.infolist():
            if not (root/item.filename).resolve().is_relative_to(root): raise ValueError('Unsafe archive path')
        zz.extractall(root)
print('Metadata saved. Parent archive is not the historical experiment manifest.')
