"""Force-platform reference events of Dataset 1 (Pacini Panebianco et al.).
Contacts: 20 N above the median baseline of each plate, platform signals at 1000 Hz on the trigger synchronization
of the release, IMU at 285.7 Hz. Each contact is assigned to the leg whose foot IMU has the smaller angular velocity.
Writes data/derived/dataset1_reference_events.csv."""
import paths as PTH
import scipy.io as sio, numpy as np, glob, re, json, pandas as pd
from scipy.signal import butter, filtfilt
R=f"{PTH.PAC}"; FS=1000/3.5; FSP=1000.; THR=20.; MINC=0.20
def contacts(fv):
    fv=fv-np.median(fv)   # remove plate baseline offset (trials 8, 26 have -325..-775 N offsets)
    A=(fv>THR).astype(int); r=np.flatnonzero(np.diff(A)==1)+1; f=np.flatnonzero(np.diff(A)==-1)+1
    return [(rr,f[f>rr][0]) for rr in r if np.any(f>rr) and (f[f>rr][0]-rr)>=MINC*FSP]
def load(t):
    imu=sio.loadmat(f'{R}/IMU data/Walking_{t}.mat',squeeze_me=True,struct_as_record=False)['Walking']
    fp=sio.loadmat(f'{R}/Synchronized Force Platform Data/Walking_Plate_{t}.mat',squeeze_me=True,struct_as_record=False)
    return imu,fp
trials=sorted(int(re.findall(r'_(\d+)\.mat',p)[0]) for p in glob.glob(R+'/Synchronized Force Platform Data/*.mat'))
rows=[]; sig={}
b,a=butter(4,15/(FS/2))
for t in trials:
    imu,fp=load(t); n=imu.rshank.gyr.shape[0]
    pol={'R':1 if np.percentile(imu.rshank.gyr[:,2],99)>-np.percentile(imu.rshank.gyr[:,2],1) else -1,
         'L':1 if np.percentile(imu.lshank.gyr[:,2],99)>-np.percentile(imu.lshank.gyr[:,2],1) else -1}
    sig[t]={'R':pol['R']*imu.rshank.gyr[:,2]*np.pi/180,'L':pol['L']*imu.lshank.gyr[:,2]*np.pi/180,'n':n,'pol':pol}
    rf_,lf_=(imu.lfoot,imu.rfoot) if t==22 else (imu.rfoot,imu.lfoot)   # trial 22: foot IMU labels swapped in the release (foot-shank correlation evidence)
    fr=np.linalg.norm(filtfilt(b,a,rf_.gyr,axis=0),axis=1); fl=np.linalg.norm(filtfilt(b,a,lf_.gyr,axis=0),axis=1)
    for p in ['footplate_1','footplate_2']:
        if p not in fp: continue
        for rr,ff in contacts(fp[p].F_V):
            hs=rr/FSP; to=ff/FSP
            if to*FS>=n-1: continue
            i0=int((hs+0.3*(to-hs))*FS); i1=int((hs+0.7*(to-hs))*FS)   # middle 40% of contact
            mr,ml=np.median(fr[i0:i1]),np.median(fl[i0:i1])
            side='R' if mr<ml else 'L'; conf=max(mr,ml)/max(min(mr,ml),1e-6)
            rows.append(dict(trial=t,plate=p,side=side,hs_s=hs,to_s=to,stance_s=to-hs,foot_ratio=conf,
                             peakF=(fp[p].F_V-np.median(fp[p].F_V))[rr:ff].max()))
G=pd.DataFrame(rows)
G.to_csv(f'{PTH.GT}',index=False)
np.save(f'{PTH.CACHE}/pacini_signals.npy',sig,allow_pickle=True)
print('contacts total',len(G),'| per side',G.side.value_counts().to_dict(),'| trials',G.trial.nunique())
print('stance s: median %.2f IQR %.2f-%.2f'%(G.stance_s.median(),*G.stance_s.quantile([.25,.75])))
print('side-assignment confidence (swing/stance foot gyro ratio): median %.1f, <2: %d contacts'%(G.foot_ratio.median(),(G.foot_ratio<2).sum()))
print(G.groupby('trial').size().to_string())
