"""MSW height rule on the development subjects (AB06-AB17) and the held-out subjects (AB18-AB30) of Dataset 2.
Runs msw_height_rule.py and writes results/heldout.json."""
import subprocess, json, re, sys
import paths as PTH
subs = ['AB06','AB07','AB08','AB09','AB10','AB11','AB12','AB13','AB14','AB15','AB16','AB17','AB18','AB19','AB20','AB21','AB23','AB24','AB25','AB27','AB28','AB30']
dev = [s for s in subs if s <= 'AB17']; test = [s for s in subs if s >= 'AB18']
H = {}
for name, ss in [('dev', dev), ('test', test)]:
    out = subprocess.run([sys.executable, 'msw_height_rule.py', ','.join(ss)], capture_output=True, text=True, check=True).stdout
    H[name] = {}
    for line in out.splitlines():
        m = re.match(r'height>=([\d.]+)\*P99:', line)
        if not m: continue
        d = H[name][m.group(1)] = {}
        for act, p, r, f in re.findall(r'\| (\w+) P ([\d.]+) R ([\d.]+) F1 ([\d.]+)', line):
            d[act] = dict(P=float(p), R=float(r), F1=float(f))
H = dict(lg0=H['test']['0.0']['level']['F1'], lg3=H['test']['0.3']['level']['F1'], st0=H['test']['0.0']['stair']['F1'], st3=H['test']['0.3']['stair']['F1'], **H)
json.dump(H, open(PTH.RESULTS + '/heldout.json', 'w'), indent=1); print({k: H[k] for k in ['lg0', 'lg3', 'st0', 'st3']})
