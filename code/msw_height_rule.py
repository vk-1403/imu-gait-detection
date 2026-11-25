"""Precision, recall and F1 of HS against gcRight for MSW height thresholds from 0 to 0.5 P99.
Usage: python msw_height_rule.py <comma-separated subjects of Dataset 2>"""
import paths as PTH
import numpy as np, pandas as pd, sys, glob, os, scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks, argrelmin
sys.path.insert(0,'.'); import metrics as M
Dd=f'{PTH.CAM}'
def detect(g, fs, hfrac):
    b,a=butter(4,15/(fs/2)); y=filtfilt(b,a,g); y=y-np.median(y)
    iqr=np.percentile(y,75)-np.percentile(y,25); amp=np.percentile(y,99)
    msw,_=find_peaks(y,prominence=0.30*iqr,distance=int(0.45*fs),height=hfrac*amp if hfrac>0 else None)
    neg,_=find_peaks(-y,prominence=0.20*iqr,distance=int(0.20*fs)); lmin=argrelmin(y)[0]; hs=[];to=[]
    for k,m in enumerate(msw):
        nxt=msw[k+1] if k+1<len(msw) else len(y); prv=msw[k-1] if k>0 else 0
        c=lmin[(lmin>m)&(lmin<nxt)&(y[lmin]<0)]
        if len(c): hs.append(c[0])
        c=neg[(neg<m)&(neg>prv)]
        if len(c): to.append(c[-1])
    return np.array(sorted(set(hs)),int),np.array(sorted(set(to)),int)
subs=sys.argv[1].split(','); res=[]
for hf in [0.0,0.2,0.3,0.4,0.5]:
    tp=nd=ng=0; per={}
    for s in subs:
        z=np.load(f'{Dd}/{s}_signals.npz'); E=pd.read_csv(f'{Dd}/{s}_events.csv')
        tm=[k for k in z.files if k.startswith('treadmill')]
        gall=np.concatenate([np.nan_to_num(z[k][:,2].astype(float)) for k in tm]); pol=1 if np.percentile(gall,99)>-np.percentile(gall,1) else -1
        for k in z.files:
            key=k.replace('__','/'); act=key.split('/')[0]; S=z[k]; t=S[:,0].astype(float)
            g0=np.nan_to_num(S[:,2].astype(float)); g0=g0 if np.percentile(g0,99)>-np.percentile(g0,1) else -g0; hs,_=detect(g0,200.,hf); hs=t[hs]
            gt=np.sort(E[(E.key==key)&(E.src=='gc')&(E.event=='HS')].t.values)
            if len(gt)<2: continue
            m=M.match(hs,gt,0.1); n_d=np.sum((hs>=gt[0]-0.3)&(hs<=gt[-1]+0.3))
            a=per.setdefault(act,[0,0,0]); a[0]+=len(m); a[1]+=n_d; a[2]+=len(gt)
    line=f'height>={hf:.1f}*P99:'
    for act,(tp,nd,ng) in sorted(per.items()):
        p=tp/nd; r=tp/ng; line+=f' | {act[:5]} P {p:.3f} R {r:.3f} F1 {2*p*r/(p+r):.3f}'
    print(line,flush=True)
