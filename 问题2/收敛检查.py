# -*- coding: utf-8 -*-
"""问题2 收敛检查：3h 时刻温度/水分对网格(N)与时间步(dt)的收敛性 + 质量守恒"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

R = 0.02; h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55; t_end = 10800.0

def rho(C):   return 650.0 + 128.0 * C
def cp(C):    return 1450.0 + 2736.0 * C / (C + 1.0)
def kcond(C): return 0.21 + 0.38 * C / (C + 1.0)
def Dfun(C, TC): return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / (TC + 273.15))

wb = openpyxl.load_workbook(os.path.join('附件', '附件1.xlsx'), data_only=True); ws = wb.active
rows = list(ws.iter_rows(values_only=True))[1:]
ta = np.array([r[0] for r in rows], float); Ta = np.array([r[1] for r in rows], float); Ca = np.array([r[2] for r in rows], float)
Tair = PchipInterpolator(ta, Ta); Cair = PchipInterpolator(ta, Ca)
def Tbc(t): return Tair(t) if t <= 14400.0 else 49.97
def Cbc(t): return Cair(t) if t <= 14400.0 else 0.0499

def thomas(a, b, c, d):
    n = len(b); cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]; cp_[i] = c[i] / m; dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1): x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def solve(N, dt):
    dr = R / N; n_steps = int(round(t_end / dt))
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    r = np.arange(N + 1) * dr; m0 = np.trapezoid(C * r, r)
    for n in range(n_steps):
        tn = (n + 1) * dt
        Tn = T.copy(); Cn = C.copy()
        for it in range(50):
            Tp = Tn.copy(); Cp = Cn.copy()
            rhocp = rho(Cp) * cp(Cp); kf = kcond(0.5 * (Cp[:-1] + Cp[1:]))
            aT = np.zeros(N + 1); bT = np.zeros(N + 1); cT = np.zeros(N + 1); dT = np.zeros(N + 1)
            l0 = kf[0] * dt / (rhocp[0] * dr**2); bT[0] = 1 + 4 * l0; cT[0] = -4 * l0; dT[0] = T[0]
            for i in range(1, N):
                lL = kf[i - 1] * dt / (rhocp[i] * dr**2); lR = kf[i] * dt / (rhocp[i] * dr**2)
                aT[i] = -(i - 0.5) / i * lL; cT[i] = -(i + 0.5) / i * lR
                bT[i] = 1 + (i - 0.5) / i * lL + (i + 0.5) / i * lR; dT[i] = T[i]
            Ah = 2 * N * h * dt / ((N - 0.25) * rhocp[N] * dr); lN = kf[N - 1] * dt / (rhocp[N] * dr**2)
            Bh = 2 * (N - 0.5) * lN / (N - 0.25)
            aT[N] = -Bh; bT[N] = 1 + Ah + Bh; dT[N] = T[N] + Ah * Tbc(tn)
            Tn = thomas(aT, bT, cT, dT)
            Cf = 0.5 * (Cp[:-1] + Cp[1:]); Tf = 0.5 * (Tn[:-1] + Tn[1:]); Df = Dfun(Cf, Tf); lam = Df * dt / dr**2
            aC = np.zeros(N + 1); bC = np.zeros(N + 1); cC = np.zeros(N + 1); dC = np.zeros(N + 1)
            bC[0] = 1 + 4 * lam[0]; cC[0] = -4 * lam[0]; dC[0] = C[0]
            for i in range(1, N):
                aC[i] = -(i - 0.5) / i * lam[i - 1]; cC[i] = -(i + 0.5) / i * lam[i]
                bC[i] = 1 + (i - 0.5) / i * lam[i - 1] + (i + 0.5) / i * lam[i]; dC[i] = C[i]
            Am = 2 * N * hm * dt / ((N - 0.25) * dr); Bm = 2 * (N - 0.5) * lam[N - 1] / (N - 0.25)
            aC[N] = -Bm; bC[N] = 1 + Am + Bm; dC[N] = C[N] + Am * Cbc(tn)
            Cn = thomas(aC, bC, cC, dC)
            if np.max(np.abs(Tn - Tp)) < 1e-6 and np.max(np.abs(Cn - Cp)) < 1e-8:
                break
        T = Tn; C = Cn
    return T[0], T[N], C[0], C[N], np.trapezoid(C * r, r) / m0

print('===== 问题2 收敛检查（3h 时刻，附录3 耦合变物性） =====')
print('网格收敛(dt=1s)：')
for N in [100, 200, 400]:
    Tc, Ts, Cc, Cs, mr = solve(N, 1.0)
    print(f'  N={N}: T中心={Tc:.4f} T表面={Ts:.4f}  C中心={Cc:.6f} C表面={Cs:.6f}  含水量比={mr:.6f}')
print('时间步收敛(N=200)：')
for dt in [1.0, 0.5, 0.25]:
    Tc, Ts, Cc, Cs, mr = solve(200, dt)
    print(f'  dt={dt}: T中心={Tc:.4f} T表面={Ts:.4f}  C中心={Cc:.6f} C表面={Cs:.6f}  含水量比={mr:.6f}')
print('完成')
