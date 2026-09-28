from pathlib import Path
import sys,json,importlib.util
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.v2.dataset import crop as current
spec=importlib.util.spec_from_file_location('preserved_crop',HERE/'before/src/v2/dataset/crop.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
stored=np.load(ROOT/'data/processed/v2/frames_m22.npy',mmap_mode='r')
cases=[]
for u,f,row in [(19,33969,108584),(55,49892,478776)]:
    d=np.load(HERE/f'crop_case_U{u}_{f}.npz')
    entry=dict(subject=u,frame=f)
    for name,mod in [('old',old),('explicit_float32',current)]:
        g,m=mod.crop_both_eyes(d['frame'],d['mesh'])
        entry[name]=dict(equal=bool(np.array_equal(g,stored[row])),width=m.crop_w_px,height=m.crop_h_px)
    assert entry['explicit_float32']['equal']
    cases.append(entry)
report=dict(numpy=np.__version__,opencv=cv2.__version__,cases=cases)
(HERE/f'crop_runtime_numpy{np.__version__}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
