# -*- coding: utf-8 -*-
"""问题1 时间收敛检查（dt=1/0.5/0.25，附录2 变物性 D(C)）"""
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

R = 0.02; rho = 820.0; cp = 2600.0; k = 0.36; h = 25.0; hm = 8e-7; D0 = 7e-9; Da = 0.89
T0 = 28.0; C0 = 2.55; t_end = 1800.0; N = 200; dr = R / N; alpha = k / (rho * cp)
def Dfun(C): return D0 * np.exp(-Da / C)

wb = openpyxl.load_workbook('附件/附件1.xlsx', data_only=True); ws = wb.active
rows = list(ws.iter_rows(values_only=True))[1:]
ta = np.array([r[0] for r in rows], float); Ta = np.array([r[1] for r in rows], float); Ca = np.array([r[2] for r in rows], float)
Tair = PchipInterpolator(ta, Ta); Cair = PchipInterpolator(ta, Ca)

def thomas(a, b, c, d):
    n = len(b); cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]; cp_[i] = c[i] / m; dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1): x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def solve(dt):
    s = alpha * dt / dr**2
    Ah = 2 * N * h * dt / ((N - 0.25) * rho * cp * dr); Bh = 2 * (N - 0.5) * s / (N - 0.25)
    aT = np.zeros(N + 1); bT = np.zeros(N + 1); cT = np.zeros(N + 1)
    bT[0] = 1 + 4 * s; cT[0] = -4 * s
    for i in range(1, N): aT[i] = -(i - 0.5) / i * s; cT[i] = -(i + 0.5) / i * s; bT[i] = 1 + 2 * s
    aT[N] = -Bh; bT[N] = 1 + Ah + Bh
    Am = 2 * N * hm * dt / ((N - 0.25) * dr)
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    for n in range(int(t_end / dt)):
        tn = (n + 1) * dt
        dT = T.copy(); dT[N] += Ah * Tair(tn); T = thomas(aT, bT, cT, dT)
        Cn = C.copy()
        for it in range(50):
            Cp = Cn.copy(); Cf = 0.5 * (Cp[:-1] + Cp[1:]); lam = Dfun(Cf) * dt / dr**2
            aC = np.zeros(N + 1); bC = np.zeros(N + 1); cC = np.zeros(N + 1); dC = np.zeros(N + 1)
            bC[0] = 1 + 4 * lam[0]; cC[0] = -4 * lam[0]; dC[0] = C[0]
            for i in range(1, N):
                aC[i] = -(i - 0.5) / i * lam[i - 1]; cC[i] = -(i + 0.5) / i * lam[i]
                bC[i] = 1 + (i - 0.5) / i * lam[i - 1] + (i + 0.5) / i * lam[i]; dC[i] = C[i]
            Bm = 2 * (N - 0.5) * lam[N - 1] / (N - 0.25)
            aC[N] = -Bm; bC[N] = 1 + Am + Bm; dC[N] = C[N] + Am * Cair(tn)
            Cn = thomas(aC, bC, cC, dC)
            if np.max(np.abs(Cn - Cp)) < 1e-12: break
        C = Cn
    return T[0], T[N], C[0], C[N]

print("===== 问题1 时间收敛检查（t=1800s） =====")
print(f"{'dt':>6} | {'中心T':>10} | {'表面T':>10} | {'中心C':>10} | {'表面C':>10}")
for dt in [1.0, 0.5, 0.25]:
    Tc, Ts, Cc, Cs = solve(dt)
    print(f"{dt:>6} | {Tc:>10.4f} | {Ts:>10.4f} | {Cc:>10.4f} | {Cs:>10.4f}")
print("（四位小数稳定即时间收敛）")
