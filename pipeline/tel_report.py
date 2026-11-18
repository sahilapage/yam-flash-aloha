"""Lag (ms) of measured joints behind the plan, tracking error, per telemetry file group.
usage: tel_report.py LABEL=file1,file2 ...   (rows: t, pos7, cmd7, plan7, q)"""
import sys, numpy as np
def lag_ms(runs, src, j):
    out = []
    for r in runs:
        a, b = np.diff(r[:, src + j]), np.diff(r[:, 1 + j])
        if a.std() < 1e-4 or b.std() < 1e-5: continue
        cs = [np.corrcoef(a[:len(a) - k], b[k:])[0, 1] for k in range(-10, 30)] if False else \
             [np.corrcoef(a[max(0,-k):len(a) - max(k,0)], b[max(k,0):len(b) - max(-k,0)])[0, 1] for k in range(-15, 30)]
        if np.all(np.isnan(cs)): continue
        out.append(range(-15, 30)[int(np.nanargmax(cs))])
    return 1000 / 60 * np.median(out) if out else float("nan")
print('%-12s %s | %s | %s' % ('config', 'lag behind demo/plan ms j1..j6', 'RMS err vs plan(t) rad j1..j6', 'max'))
for arg in sys.argv[1:]:
    label, files = arg.split('=')
    runs = [r for r in (np.load(f) for f in files.split(',')) if len(r) > 60]
    lags = [lag_ms(runs, 15, j) for j in range(6)]
    e = np.vstack([r[:, 1:7] - r[:, 15:21] for r in runs])
    print('%-12s %s | %s | %.3f' % (label, ' '.join('%4.0f' % l for l in lags),
          ' '.join('%.3f' % v for v in np.sqrt((e ** 2).mean(0))), np.abs(e).max()))
