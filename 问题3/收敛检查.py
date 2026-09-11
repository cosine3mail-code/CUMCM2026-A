# -*- coding: utf-8 -*-
"""问题3 收敛检查：烘干时长对网格(N)与时间步(dt)的收敛性"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

R = 0.02; h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55; C_target = 0.15

def rho(C):   return 650.0 + 128.0 * C
def cp(C):    return 1450.0 + 2736.0 * C / (C + 1.0)
def kcond(C): return 0.21 + 0.38 * C / (C + 1.0)
def Dfun(C, TC): return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / (TC + 273.15))

wb = openpyxl.load_workbook(os.path.join('附件', '附件1.xlsx'), data_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))[1:]
ta = np.array([r[0] for r in rows], float)
Ta = np.array([r[1] for r in rows], float)
Ca = np.array([r[2] for r in rows], float)
Tair = PchipInterpolator(ta, Ta); Cair = PchipInterpolator(ta, Ca)

def Tbc(t): return Tair(t) if t <= 14400.0 else 50.0
def Cbc(t): return Cair(t) if t <= 14400.0 else 0.05

def thomas(a, b, c, d):
    n = len(b)
    cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]
        cp_[i] = c[i] / m
        dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def drying_time(N, dt):
    dr = R / N; nmax = int(500000.0 / dt)
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    c_center_prev = C0
    for n in range(nmax):
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
        if C[0] < C_target:
            return (n + (C_target - c_center_prev) / (C[0] - c_center_prev)) * dt / 3600.0
        c_center_prev = C[0]
    return None

print('烘干时长收敛性检查：')
print('网格收敛(dt=60s):')
for N in [100, 200, 400]:
    td = drying_time(N, 60.0)
    print(f'  N={N}: 烘干时长 = {td:.4f} h')
print('时间步收敛(N=200):')
for dt in [60.0, 30.0, 15.0]:
    td = drying_time(200, dt)
    print(f'  dt={dt}: 烘干时长 = {td:.4f} h')
print('完成')
