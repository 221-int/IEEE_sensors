from pathlib import Path
import sys,json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
import numpy as np
import torch,cv2
from src.v2.model import encoder as E
from src.v2.train_encoder import Bundle
from src.v2.common import splits,repro
repro.seal(0)
torch.set_num_threads(2)
b=Bundle(str(ROOT/'data/processed/v2'))
m=splits.subject_masks(b.subject,splits.load_folds(),splits.fold_rotation(0))
ids=np.flatnonzero(m['test'])[:32]; x,mask,_=b.batch(ids)
checks=[]
for stem,front in [('train_encoder_final',E.build()),('train_image_cnn_head_final',E.build_image_cnn(16))]:
    folder=ROOT/f'models/v2/{stem}/fold0_seed0';head=E.build_head()
    front.load_state_dict(torch.load(folder/'encoder.pt',map_location='cpu',weights_only=True))
    head.load_state_dict(torch.load(folder/'head.pt',map_location='cpu',weights_only=True))
    front.cuda().eval();head.cuda().eval()
    with torch.no_grad(): score=head(front(torch.from_numpy(x.reshape(-1,1,64,160)).cuda()).reshape(32,19,16),torch.from_numpy(mask).cuda()).cpu().numpy()
    expected=np.load(ROOT/f'results/v2/{stem}_scores/fold0_seed0.npz')['ours'][:32]
    error=float(abs(score-expected).max());assert error<1e-3,error
    checks.append(dict(model=stem,source_prediction_max_absolute_error=error,n_events=32))
data=json.loads((OUT/'results/external_pilot.json').read_text())
summary={}
for variant in ['ours','image_head','ear_head','ear_rule']:
    rs=[r for r in data['runs'] if r['variant']==variant]
    fields={}
    for name,fn in [('window_AP',lambda r:r['event_window']['ap_scorable']),
                    ('window_recall',lambda r:r['event_window']['recall']),
                    ('event_recall',lambda r:r['continuous']['0']['recall']),
                    ('event_F1',lambda r:r['continuous']['0']['f1']),
                    ('false_events_per_minute',lambda r:r['continuous']['0']['false_detections_per_minute']),
                    ('iou01_event_F1',lambda r:r['continuous']['0.1']['f1']),
                    ('negative_frame_positive_rate',lambda r:r['negative_frame_positive_rate']),
                    ('decision_coverage',lambda r:r['decision_coverage']),
                    ('run_median_alert_delay_s',lambda r:r['continuous']['0']['alert_delay_seconds_median'])]:
        a=np.array([fn(r) for r in rs]);fields[name]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()))
    summary[variant]=fields
(OUT/'results/external_summary.json').write_text(json.dumps(dict(summary=summary,source_checkpoint_checks=checks,scope='one video, descriptive 15-source-run variation, not population CI'),indent=2))
# Visual annotation/frame audit: fully-closed annotations plus surrounding samples.
video=ROOT/'data/_legacy_public/eyeblink8/eyeblink8/9/27122013_152435_cam.avi'
ann=[]
for l in video.with_suffix('.tag').read_text().splitlines():
    q=l.split(':')
    if len(q)==19 and q[0].isdigit():ann.append(q)
rows=np.array(ann);closed=np.flatnonzero((rows[:,3]=='C') & (rows[:,5]=='C'))
chosen=[0]+closed[np.linspace(0,len(closed)-1,7).astype(int)].tolist()
cap=cv2.VideoCapture(str(video));tiles=[]
for k in chosen:
    r=rows[k];fid=int(r[0]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
    for x,y in np.array(r[11:],int).reshape(4,2):cv2.circle(im,(x,y),3,(0,255,0),1)
    cv2.putText(im,f'frame {fid}, blink {r[1]}, closed {r[3]}/{r[5]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),1)
    tiles.append(cv2.resize(im,(320,240)))
cap.release()
cv2.imwrite(str(OUT/'results/annotation_alignment.png'),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
print(json.dumps(dict(checks=checks,summary=summary),indent=2))
