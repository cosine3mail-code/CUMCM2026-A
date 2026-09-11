# -*- coding: utf-8 -*-
"""问题3 恒温段边界 T∞/C∞ 敏感性扫描
扫描附件1 噪声范围内(±0.25℃)的恒温段取值对烘干时长的影响。"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

R = 0.02; h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55; C_target = 0.15
N = 200; dr = R / N; dt = 60.0; nmax = int(500000.0 / dt)

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

def thomas(a, b, c, d):
    n = len(b); cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]; cp_[i] = c[i] / m; dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1): x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def drying_time(Tinf, Cinf):
    def Tbc(t): return Tair(t) if t <= 14400.0 else Tinf
    def Cbc(t): return Cair(t) if t <= 14400.0 else Cinf
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

print('=== 恒温段 T∞/C∞ 敏感性扫描（问题3 烘干时长，N=200, dt=60s） ===\n')

print('【1】T∞ 扫描（固定 C∞=0.0499）：')
Tinf_vals = [49.76, 49.94, 49.97, 50.165, 50.25]
for Tv in Tinf_vals:
    td = drying_time(Tv, 0.0499)
    print(f'   T∞={Tv:6.2f}℃  ->  烘干时长 {td:.4f} h')

print('\n【2】C∞ 扫描（固定 T∞=49.97）：')
Cinf_vals = [0.0489, 0.0495, 0.0499, 0.0503]
for Cv in Cinf_vals:
    td = drying_time(49.97, Cv)
    print(f'   C∞={Cv:.4f} kg/kg  ->  烘干时长 {td:.4f} h')

print('\n【3】关键组合点：')
key_pts = [('平台均值', 49.97, 0.0499), ('平台中位', 49.94, 0.04989),
           ('末值(噪声离群点)', 50.165, 0.04986), ('平台均值（本文）', 49.97, 0.0499)]
for name, Tv, Cv in key_pts:
    td = drying_time(Tv, Cv)
    print(f'   {name:14s} (T∞={Tv}, C∞={Cv})  ->  烘干时长 {td:.4f} h')

print('\n完成')
