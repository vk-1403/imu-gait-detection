"""Figures of the paper (printed sizes).
Fig. 1 method, Fig. 2 comparison of all methods, Fig. 3 walking speed, Fig. 4 matching tolerance,
Fig. 5 subject-level robustness, Fig. 6 gait-phase durations and timing-precision index. Written to results/figures/."""
import paths as PTH
import os, sys, json, numpy as np, pandas as pd, scipy.io as sio
sys.path.insert(0, f'{PTH.CODE}')
from plot_style import *
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr
import peak_anchored_detector as D
OUT = f'{PTH.FIG}'; R = f'{PTH.RESULTS}'
SH, OTHER, IMU_C = '#4472C4', '#A6A6A6', '#FFC000'
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
                     'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.linewidth': 0.8, 'lines.linewidth': 1.1})

def seg(ax, p, q, col, lw=4.5): ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, solid_capstyle='round', zorder=2)
def joint(ax, p): ax.add_patch(Circle(p, 0.06, fc='white', ec='black', lw=0.9, zorder=5))
def imu(ax, c, ang, w=0.2, h=0.2): ax.add_patch(Rectangle((c[0] - w / 2, c[1] - h / 2), w, h, angle=ang, rotation_point='center', fc=IMU_C, ec='black', lw=0.8, zorder=6))
def along(p, q, f): return (p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]))

# ======================= Fig. 1
fig = plt.figure(figsize=(7.16, 2.45))
# (a) stick figures
ax = fig.add_axes([0.0, 0.10, 0.27, 0.88]); ax.set_xlim(-0.2, 4.3); ax.set_ylim(-0.25, 4.2); ax.set_aspect('equal'); ax.axis('off')
for x0, stance, title in [(0.0, True, 'Stance'), (2.25, False, 'Swing')]:
    pel = (x0 + 0.9, 3.6); ax.plot([x0 + 0.5, x0 + 1.3], [3.6, 3.6], color='black', lw=5.5, solid_capstyle='round')
    if stance:
        knee, ank, toe = (x0 + 0.95, 2.15), (x0 + 0.75, 0.45), (x0 + 1.25, 0.32)
        ax.plot([x0 + 0.2, x0 + 1.6], [0.25, 0.25], 'k', lw=0.9)
        for xx in np.arange(x0 + 0.25, x0 + 1.6, 0.14): ax.plot([xx, xx - 0.1], [0.25, 0.13], 'k', lw=0.6)
    else:
        knee, ank, toe = (x0 + 1.45, 2.3), (x0 + 0.75, 1.05), (x0 + 1.15, 0.75)
    seg(ax, pel, knee, OTHER); seg(ax, knee, ank, SH); seg(ax, ank, toe, OTHER, 4)
    for p in (pel, knee, ank): joint(ax, p)
    ang = np.degrees(np.arctan2(ank[1] - knee[1], ank[0] - knee[0])) + 90
    nrm = np.array([ank[1] - knee[1], -(ank[0] - knee[0])]); nrm /= np.linalg.norm(nrm); nrm = nrm if nrm[0] > 0 else -nrm
    pac = along(knee, ank, 0.88); cam = along(knee, ank, 0.70); camc = (cam[0] + 0.12 * nrm[0], cam[1] + 0.12 * nrm[1])
    imu(ax, pac, ang); imu(ax, camc, ang, 0.13, 0.22)
    ax.text(x0 + 0.9, 3.95, title, ha='center', fontsize=8)
    if stance:
        ax.annotate('D2', xy=camc, xytext=(x0 + 1.55, 1.35), fontsize=7.5, va='center', arrowprops=dict(arrowstyle='-', lw=0.6))
        ax.annotate('D1', xy=pac, xytext=(x0 + 1.55, 0.75), fontsize=7.5, va='center', arrowprops=dict(arrowstyle='-', lw=0.6))
a = (2.25 + 0.75, 1.05)
ax.add_patch(FancyArrowPatch((a[0] - 0.35, a[1] + 0.05), (a[0] + 0.3, a[1] - 0.5), connectionstyle='arc3,rad=0.4', arrowstyle='-|>', mutation_scale=8, lw=1.1, color=RED, zorder=7))
ax.text(a[0] + 0.38, a[1] - 0.45, r'$\omega>0$', color=RED, fontsize=7.5, va='top')
ax.add_patch(Rectangle((0.05, -0.2), 0.18, 0.18, fc=IMU_C, ec='black', lw=0.7)); ax.text(0.3, -0.11, 'IMU', fontsize=6.5, va='center')
fig.text(0.135, -0.03, '(a)', ha='center', fontsize=8)
# (b) flowchart (structure only; parameter values are given in the text)
ax = fig.add_axes([0.275, 0.10, 0.205, 0.88]); ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis('off')
steps = [('Shank angular velocity', r'sagittal $\omega(t)$'), ('Polarity check', 'mid-swing peak positive'),
         ('Low-pass + offset', 'zero-phase Butterworth; median'), ('Mid-swing anchors', 'prominent, high, spaced peaks'),
         ('Toe-off', 'last negative peak before anchor'), ('Heel strike', 'first minimum < 0 after anchor'), ('Output', r'HS, TO, MSW and $\sigma_t$')]
ax.add_patch(FancyBboxPatch((1.75, 9.75), 8.15, 0.6, boxstyle='round,pad=0,rounding_size=0.15', fc=GOLD, ec='black', lw=0.7))
ax.text(5.82, 10.05, 'Peak-anchored detector', ha='center', va='center', weight='bold', fontsize=7.2)
H, G = 0.95, 0.50; ys = []
for i, (h, s) in enumerate(steps):
    y = 9.75 - (i + 1) * (H + G) + G - 0.05; ys.append(y)
    ax.add_patch(Rectangle((1.75, y), 8.15, H, fc='white', ec='black', lw=0.7))
    ax.text(5.82, y + 0.66, h, ha='center', va='center', weight='bold', fontsize=6.8)
    ax.text(5.82, y + 0.25, s, ha='center', va='center', fontsize=6.2)
    if i < len(steps) - 1: ax.annotate('', xy=(5.82, y - G), xytext=(5.82, y), arrowprops=dict(arrowstyle='-|>', color='black', lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
for lab, i0, i1 in [('Pre-processing', 0, 2), ('Event rules', 3, 5), ('Out', 6, 6)]:
    y0, y1 = ys[i1], ys[i0] + H
    ax.add_patch(FancyBboxPatch((0.15, y0), 1.25, y1 - y0, boxstyle='round,pad=0,rounding_size=0.3', fc=LIGHTBLUE, ec='black', lw=0.7))
    ax.text(0.78, (y0 + y1) / 2, lab, rotation=90, ha='center', va='center', weight='bold', fontsize=6.5)
fig.text(0.378, -0.03, '(b)', ha='center', fontsize=8)
# (c) representative cycle
PAC = f'{PTH.PAC}'; FS = 1000 / 3.5; t_sub = 6
imu_ = sio.loadmat(f'{PAC}/IMU data/Walking_{t_sub}.mat', squeeze_me=True, struct_as_record=False)['Walking']
fp = sio.loadmat(f'{PAC}/Synchronized Force Platform Data/Walking_Plate_{t_sub}.mat', squeeze_me=True, struct_as_record=False)
g = imu_.rshank.gyr[:, 2] * np.pi / 180; g = g if np.percentile(g, 99) > -np.percentile(g, 1) else -g
d = D.detect(g, FS, return_all=True); y = d['y']; tt = np.arange(len(y)) / FS
Gt = pd.read_csv(f'{PTH.GT}'); Gt = Gt[(Gt.trial == t_sub) & (Gt.side == 'R') & (Gt.peakF >= 300)].sort_values('hs_s')
c0 = Gt[Gt.stance_s.between(0.69, 0.74)].iloc[1]; w0, w1 = c0.hs_s - 1.15, c0.hs_s + 1.85
a1 = fig.add_axes([0.555, 0.43, 0.435, 0.48]); a2 = fig.add_axes([0.555, 0.17, 0.435, 0.22], sharex=a1)
m = (tt >= w0) & (tt <= w1); a1.plot(tt[m], y[m], color=BLUE, lw=1.1); a1.axhline(0, color=GRAY, lw=0.5, ls='--')
for key, mk, col, lab in [('msw', '^', GREEN, 'MSW'), ('to', 'v', RED, 'TO'), ('hs', 'o', MAGENTA, 'HS')]:
    k = d[key].astype(int); k = k[(tt[k] >= w0) & (tt[k] <= w1)]; a1.plot(tt[k], y[k], mk, ms=4.5, color=col, mec='black', mew=0.4, label=lab, zorder=5)
hs_k = d['hs'][(d['hs'] / FS >= c0.hs_s - 0.15)][0] / FS; to_k = d['to'][d['to'] / FS > hs_k][0] / FS; hs2 = d['hs'][d['hs'] / FS > to_k][0] / FS
yy = y[m].min() - 0.4
for x0_, x1_, lab in [(hs_k, to_k, f'stance {1000 * (to_k - hs_k):.0f} ms'), (to_k, hs2, f'swing {1000 * (hs2 - to_k):.0f} ms')]:
    a1.annotate('', xy=(x0_ + 0.01, yy), xytext=(x1_ - 0.01, yy), arrowprops=dict(arrowstyle='<->', lw=0.6))
    a1.text((x0_ + x1_) / 2, yy - 0.25, lab, ha='center', va='top', fontsize=6.3)
a1.set_ylim(y[m].min() - 2.3, y[m].max() + 0.8); a1.set_ylabel(r'$\omega$ (rad/s)')
a1.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
plt.setp(a1.get_xticklabels(), visible=False)
tf = np.arange(len(fp['footplate_1'].F_V)) / 1000.0; mf = (tf >= w0) & (tf <= w1)
for p_, col in [('footplate_1', BLACK), ('footplate_2', OLIVE)]:
    Fv = fp[p_].F_V - np.median(fp[p_].F_V); a2.plot(tf[mf], Fv[mf] / 100, color=col, lw=0.9)
a2.axhline(0.2, color=RED, lw=0.6, ls='--')
for _, r in Gt[(Gt.hs_s > w0) & (Gt.to_s < w1)].iterrows():
    a2.axvspan(r.hs_s, r.to_s, color='#d9d9d9', lw=0)
    a2.text((r.hs_s + r.to_s) / 2, 7.6, f'contact {1000 * r.stance_s:.0f} ms', ha='center', fontsize=6.3)
    for a_ in (a1, a2): a_.axvline(r.hs_s, color=MAGENTA, lw=0.5, ls=':'); a_.axvline(r.to_s, color=RED, lw=0.5, ls=':')
a2.set_ylim(-0.5, 9.5); a2.set_ylabel('GRF\n(100 N)'); a2.set_xlabel('Time (s)')
fig.text(0.77, -0.03, '(c)', ha='center', fontsize=8)
fig.savefig(f'{OUT}/Fig1_method.png'); fig.savefig(f'{OUT}/Fig1_method.pdf'); plt.close(fig)

# ======================= Fig. 2
import re as _re
_BIB = dict(Aminian2002='9', Salarian2004='10', Greene2010='11', Catalfamo2010='12')   # reference numbers in the paper
if os.path.exists(os.environ.get('PAPER_AUX', '')): _BIB = dict(_re.findall(r'\\bibcite\{([^}]*)\}\{([^}]*)\}', open(os.environ['PAPER_AUX']).read()))
def _cite(name, key): return f'{name} et al. [{_BIB.get(key, "?")}]'
NAMES = dict(peak_anchored='Proposed', rf='RF (trained)', hgb='HGB (trained)', cnn='TCN (trained)',
             salarian2004=_cite('Salarian', 'Salarian2004'), catalfamo2010=_cite('Catalfamo', 'Catalfamo2010'),
             greene2010=_cite('Greene', 'Greene2010'), aminian2002=_cite('Aminian', 'Aminian2002'), fixed_threshold='Fixed threshold')
COL = dict(peak_anchored=BLUE, rf=RED, hgb=MAGENTA, cnn=GREEN, salarian2004=OLIVE, catalfamo2010=ORANGE, greene2010=CYAN, aminian2002=VIOLET, fixed_threshold=GRAY)
MK = dict(peak_anchored='o', rf='s', hgb='D', cnn='h', salarian2004='^', catalfamo2010='v', greene2010='<', aminian2002='>', fixed_threshold='p')
S = pd.read_csv(f'{R}/stats_subject_level.csv'); rec = S[S.metric == 'recall']
order = list(rec.groupby('detector')['mean'].min().sort_values(ascending=False).index)
panels = [('Pacini', 'HS', 'Dataset 1, heel strike'), ('Pacini', 'TO', 'Dataset 1, toe-off'), ('Camargo', 'HS', 'Dataset 2, heel strike'), ('Camargo', 'TO', 'Dataset 2, toe-off')]
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.35), sharey=True, gridspec_kw=dict(wspace=0.12, hspace=0.42, height_ratios=[1, 1.05]))
ev_cache = {}
for j, (ds, ev, title) in enumerate(panels):
    ax = axs[0, j]; dd = rec[(rec.dataset == ds) & (rec.ref == 'FP') & (rec.event == ev)].set_index('detector')
    for i, det in enumerate(order):
        r = dd.loc[det]; yy = len(order) - 1 - i
        ax.errorbar(100 * r['mean'], yy, xerr=[[100 * (r['mean'] - r.ci_lo)], [100 * (r.ci_hi - r['mean'])]], fmt=MK[det], color=COL[det],
                    mec='black', mew=0.4, ms=4.5, capsize=1.5, elinewidth=0.8, lw=0)
    ax.axvline(100 * dd.loc['peak_anchored', 'mean'], color=BLUE, lw=0.5, ls='--', zorder=0)
    ax.set_xlim(0, 103); ax.set_xticks([0, 50, 100]); ax.set_title(title, fontsize=7.5, pad=3)
    if j == 0: ax.set_xlabel('Recall at ±100 ms (%)'); ax.xaxis.set_label_coords(2.15, -0.17)
    ax = axs[1, j]; data = []
    for det in order:
        key = (det, ds)
        if key not in ev_cache: ev_cache[key] = pd.read_csv(f'{R}/{det}/{"pacini" if ds == "Pacini" else "camargo"}_events.csv')
        P = ev_cache[key]; data.append(P[(P.ref == 'FP') & (P.event == ev) & (P.tol == 150) & P.matched].err_ms.values)
    bp = ax.boxplot(data[::-1], positions=range(len(order)), vert=False, whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', lw=0.8), boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, det in zip(bp['boxes'], order[::-1]): patch.set_facecolor(COL[det]); patch.set_alpha(0.6)
    ax.axvline(0, color='black', lw=0.5, ls='--'); ax.set_xlim(-150, 150); ax.set_xticks([-100, 0, 100])
    if j == 0: ax.set_xlabel('IMU − force platform (ms)'); ax.xaxis.set_label_coords(2.15, -0.17)
for r_ in range(2):
    axs[r_, 0].set_yticks(range(len(order))); axs[r_, 0].set_yticklabels([NAMES[o] for o in order][::-1]); axs[r_, 0].set_ylim(-0.6, len(order) - 0.4)
fig.text(0.005, 0.93, '(a)', fontsize=8); fig.text(0.005, 0.47, '(b)', fontsize=8)
fig.savefig(f'{OUT}/Fig2_comparison.png'); fig.savefig(f'{OUT}/Fig2_comparison.pdf'); plt.close(fig)

# ======================= Fig. 3
Ph = pd.read_csv(f'{R}/phase_durations.csv'); E = pd.read_csv(f'{R}/matched_errors_sigma.csv').dropna()
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.2), gridspec_kw=dict(hspace=0.78, wspace=0.42))
for ax, (ds, act, prm, title), lab in zip(axs[0], [('Pacini', 'levelground', 'stance', 'Dataset 1 stance'), ('Camargo', 'treadmill', 'stride', 'Dataset 2 treadmill stride')], ['(a)', '(b)']):
    g = Ph[(Ph.ds == ds) & (Ph.act == act) & (Ph.param == prm)]
    ax.scatter(g.ref_ms + g.err_ms / 2, g.err_ms, s=1.5, color=BLUE, alpha=0.3, edgecolor='none', rasterized=True)
    b, s_ = g.err_ms.mean(), g.err_ms.std()
    for v, ls in [(b, '-'), (b - 1.96 * s_, '--'), (b + 1.96 * s_, '--')]: ax.axhline(v, color=RED, lw=0.8, ls=ls)
    ax.set_title(title, fontsize=7.5, pad=3); ax.set_ylim(-200, 200); ax.set_xlabel('Mean duration (ms)'); ax.set_ylabel('IMU − FP (ms)')
    ax.text(0.5, -0.5, lab, transform=ax.transAxes, ha='center', fontsize=8)
for ax, ev, col, lab in zip(axs[1], ['HS', 'TO'], [MAGENTA, RED], ['(c)', '(d)']):
    g = E[(E.ds == 'Pacini') & (E.ev == ev)]
    ax.scatter(g.sigma_ms, g.err_ms.abs().clip(lower=0.3), s=2.5, color=col, alpha=0.5, edgecolor='none', rasterized=True)
    ax.plot([0.1, 300], [0.1, 300], 'k--', lw=0.6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.1, 300); ax.set_ylim(0.3, 300)
    ax.set_title('Dataset 1 heel strike' if ev == 'HS' else 'Dataset 1 toe-off', fontsize=7.5, pad=3)
    ax.set_xlabel(r'$\sigma_t$ (ms)'); ax.set_ylabel('|error| (ms)')
    ax.text(0.5, -0.5, lab, transform=ax.transAxes, ha='center', fontsize=8)
fig.savefig(f'{OUT}/Fig6_timing.png'); fig.savefig(f'{OUT}/Fig6_timing.pdf'); plt.close(fig)
print('ok', order)

# ======================= Fig. 4: recall and timing error vs treadmill speed
Rs = pd.read_csv(f'{R}/speed_events.csv')
edges = [0.45, 0.75, 1.05, 1.35, 1.65, 1.95]; mids = [0.6, 0.9, 1.2, 1.5, 1.8]
Rs['bin'] = pd.cut(Rs.speed, edges, labels=mids).astype(float)
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.45), sharex=True, gridspec_kw=dict(wspace=0.12, hspace=0.3))
for j, (ev, title) in enumerate([('HS', 'Heel strike'), ('TO', 'Toe-off')]):
    for det in order:
        g = Rs[(Rs.detector == det) & (Rs.event == ev)]
        r_ = g.groupby(['bin', 'subject']).matched.mean().groupby('bin').mean()
        e_ = g[g.err_ms.notna()].groupby('bin').err_ms.median()
        kw = dict(marker=MK[det], color=COL[det], mec='black', mew=0.3, ms=3.2, lw=1.4 if det == 'peak_anchored' else 0.8, zorder=5 if det == 'peak_anchored' else 3)
        axs[0, j].plot(r_.index, 100 * r_.values, label=NAMES[det], **kw); axs[1, j].plot(e_.index, e_.values, **kw)
    axs[0, j].set_title(title, fontsize=7.5, pad=3); axs[0, j].set_ylim(0, 103)
    axs[1, j].axhline(0, color='black', lw=0.5, ls='--'); axs[1, j].set_ylim(-70, 150); axs[1, j].set_xlabel('Belt speed (m/s)')
    axs[1, j].set_xticks(mids); axs[1, j].set_xticklabels(['0.6', '0.9', '1.2', '1.5', '1.8'])
    if j == 1: plt.setp(axs[0, j].get_yticklabels(), visible=False); plt.setp(axs[1, j].get_yticklabels(), visible=False)
axs[0, 0].set_ylabel('Recall (%)'); axs[1, 0].set_ylabel('Median error (ms)')
for ax, l, tt_ in zip(axs.flat, ['(a)', '(b)', '(c)', '(d)'], ['Heel strike', 'Toe-off', 'Heel strike', 'Toe-off']): ax.set_title(f'{l} {tt_}', fontsize=7.5, pad=3)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.885), fontsize=6, frameon=False, handletextpad=0.3, columnspacing=0.7)
fig.subplots_adjust(top=0.83)
fig.savefig(f'{OUT}/Fig3_speed.png'); fig.savefig(f'{OUT}/Fig3_speed.pdf'); plt.close(fig)
print('fig4 ok')

# ======================= Fig. 5: recall vs matching tolerance, all methods
PS = pd.read_csv(f'{R}/per_subject_all.csv'); PS = PS[(PS.ref == 'FP') & PS.detector.isin(order)]
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.45), sharex=True, sharey=True, gridspec_kw=dict(wspace=0.1, hspace=0.32))
for ax, (ds, ev, title) in zip(axs.flat, [('Pacini', 'HS', '(a) Dataset 1, heel strike'), ('Pacini', 'TO', '(b) Dataset 1, toe-off'),
                                         ('Camargo', 'HS', '(c) Dataset 2, heel strike'), ('Camargo', 'TO', '(d) Dataset 2, toe-off')]):
    d = PS[(PS.dataset == ds) & (PS.event == ev)]
    m = d.groupby(['detector', 'tol', 'subject']).recall.mean().groupby(['detector', 'tol']).mean()
    for det in order:
        kw = dict(marker=MK[det], color=COL[det], mec='black', mew=0.3, ms=3.2, lw=1.4 if det == 'peak_anchored' else 0.8,
                  zorder=5 if det == 'peak_anchored' else 3)
        ax.plot([25, 50, 100, 150], 100 * m.loc[det].reindex([25, 50, 100, 150]).values, label=NAMES[det], **kw)
    ax.set_title(title, fontsize=7.5, pad=3); ax.set_ylim(0, 103); ax.set_xticks([25, 50, 100, 150])
for ax in axs[1]: ax.set_xlabel('Tolerance (± ms)')
for ax in axs[:, 0]: ax.set_ylabel('Recall (%)')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.885), fontsize=6, frameon=False, handletextpad=0.3, columnspacing=0.7)
fig.subplots_adjust(top=0.83)
fig.savefig(f'{OUT}/Fig4_tolerance.png'); fig.savefig(f'{OUT}/Fig4_tolerance.pdf'); plt.close(fig)

# ======================= Fig. 6: subject-level robustness (lowest of HS/TO recall per subject)
PS2 = pd.read_csv(f'{R}/per_subject_all.csv'); PS2 = PS2[(PS2.ref == 'FP') & (PS2.tol == 100) & PS2.detector.isin(order)]
W = PS2.groupby(['detector', 'dataset', 'subject', 'event']).recall.mean().unstack('event').min(axis=1)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.85), sharey=True, gridspec_kw=dict(wspace=0.08))
rng_ = np.random.default_rng(0)
for ax, ds, title in zip(axs, ['Pacini', 'Camargo'], ['(a) Dataset 1, 27 subjects', '(b) Dataset 2, 22 subjects']):
    for i, det in enumerate(order):
        v = 100 * W.loc[det, ds].values; yy = len(order) - 1 - i
        ax.scatter(v, yy + rng_.uniform(-0.18, 0.18, len(v)), s=4, color=COL[det], edgecolor='black', lw=0.2, alpha=0.8, zorder=3)
        ax.plot([np.median(v)] * 2, [yy - 0.32, yy + 0.32], color='black', lw=1.2, zorder=4)
        ax.text(111, yy, f'{int((v < 90).sum())}', va='center', ha='center', fontsize=6)
    ax.axvline(90, color='black', lw=0.5, ls='--', zorder=1); ax.axvline(102, color='black', lw=0.6, zorder=1); ax.set_xlim(0, 120); ax.set_xticks([0, 50, 90])
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.text(111, len(order) - 0.45, '<90%', fontsize=5.5, ha='center', va='center')
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels([NAMES[o] for o in order][::-1], fontsize=6.5); axs[0].set_ylim(-0.6, len(order) - 0.05)
fig.text(0.62, 0.025, 'Lowest of HS and TO recall per subject (%)', ha='center', fontsize=7)
fig.subplots_adjust(left=0.27, right=0.98, bottom=0.18, top=0.89)
fig.savefig(f'{OUT}/Fig5_subjects.png'); fig.savefig(f'{OUT}/Fig5_subjects.pdf'); plt.close(fig)
