"""Collects the per-event results of all methods into results/summary_all.csv and results/per_subject_all.csv."""
import paths as PTH
import pandas as pd, numpy as np, sys
sys.path.insert(0,f'{PTH.CODE}'); import metrics as M
B=f'{PTH.RESULTS}'; DETS=['peak_anchored','catalfamo2010','salarian2004','greene2010','aminian2002','hgb','rf','cnn','fixed_threshold']
def load(d):
    P=pd.read_csv(f'{B}/{d}/pacini_events.csv'); C=pd.read_csv(f'{B}/{d}/camargo_events.csv'); PR=pd.read_csv(f'{B}/{d}/camargo_precision.csv')
    return pd.concat([P,C]),PR
summ=[]; subj=[]
for d in DETS:
    A,PR=load(d)
    for (ds,ref,act,ev,tol),g in A.groupby(['dataset','ref','activity','event','tol']):
        ps=g.groupby('subject').matched.mean()
        es=g[g.matched].groupby('subject').err_ms
        e=g[g.matched].err_ms
        S=M.summarise_errors(e)
        row=dict(detector=d,dataset=ds,ref=ref,activity=act,event=ev,tol=tol,n_events=len(g),n_subj=g.subject.nunique(),
                 recall_pooled=g.matched.mean(),recall_subj_mean=ps.mean(),recall_subj_sd=ps.std(),
                 recall_ci=M.boot_ci(ps.values),bias=S['bias'],sd=S['sd'],mae=S['mae'],loa_lo=S['loa_lo'],loa_hi=S['loa_hi'])
        if ref=='gc':
            q=PR[(PR.activity==act)&(PR.event==ev)&(PR.tol==tol)].groupby('subject')[['n_det','n_gt','tp']].sum()
            pr=q.tp/q.n_det; rc=q.tp/q.n_gt; f1=2*pr*rc/(pr+rc)
            row.update(precision_subj_mean=pr.mean(),f1_subj_mean=f1.mean(),f1_subj_sd=f1.std(),det_per_gt=(q.n_det/q.n_gt).median())
            for s_,v in f1.items(): subj.append(dict(detector=d,dataset=ds,ref=ref,activity=act,event=ev,tol=tol,subject=s_,f1=v,recall=rc[s_],precision=pr[s_]))
        for s_,v in ps.items():
            subj.append(dict(detector=d,dataset=ds,ref=ref,activity=act,event=ev,tol=tol,subject=s_,recall=v,mae=es.apply(lambda x:np.abs(x).mean()).get(s_,np.nan)))
        summ.append(row)
S=pd.DataFrame(summ); S.to_csv(f'{B}/summary_all.csv',index=False); pd.DataFrame(subj).to_csv(f'{B}/per_subject_all.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
v=S[(S.tol==100)&(S.detector=='peak_anchored')][['dataset','ref','activity','event','n_events','n_subj','recall_pooled','recall_subj_mean','recall_subj_sd','bias','sd','mae','precision_subj_mean','f1_subj_mean','det_per_gt']]
print(v.round(3).to_string(index=False))
print()
w=S[(S.tol==100)&(S.event=='HS')].pivot_table(index=['dataset','ref','activity'],columns='detector',values='recall_subj_mean').round(3)
print('HS recall @±100 ms (subject mean)'); print(w.to_string())
w=S[(S.tol==100)&(S.event=='TO')].pivot_table(index=['dataset','ref','activity'],columns='detector',values='recall_subj_mean').round(3)
print('TO recall @±100 ms (subject mean)'); print(w.to_string())
