# -*- coding: utf-8 -*-
"""问题4 路线B(非均匀收缩) 收敛检查"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55; C_target = 0.15
def rho(C):   return 760.0 + 90.0 * C
def cp(C):    return 1850.0 + 2150.0 * C / (C + 1.0)
def kcond(C): return 0.12 + 0.20 * C / (C + 1.0)
def Dfun(C, TC): return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / (TC + 273.15))
def vfun(C):  return (1.0 + C) / rho(C)

def read(path, ncol):
    wb = openpyxl.load_workbook(path, data_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    return tuple(np.array([r[j] for r in rows], float) for j in range(ncol))

t_air, T_air, C_air = read(os.path.join('附件', '附件1.xlsx'), 3)
Tair = PchipInterpolator(t_air, T_air); Cair = PchipInterpolator(t_air, C_air)
t_rad, R_rad = read(os.path.join('附件', '附件2.xlsx'), 2)
R_rad = R_rad / 100.0
Rp = PchipInterpolator(t_rad, R_rad); R_min = float(R_rad[-1])
def Tbc(t): return Tair(t) if t <= 14400.0 else 50.0
def Cbc(t): return Cair(t) if t <= 14400.0 else 0.05
def R_of(t): return R_min if t >= t_rad[-1] else float(Rp(t))

def thomas(a, b, c, d):
    n = len(b); cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]; cp_[i] = c[i] / m; dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1): x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def drying_time(N, dt):
    nmax = int(500000.0 / dt)
    s = (np.arange(N + 1) / N) ** 2
    s_face = 0.5 * (s[:-1] + s[1:])
    s_lo = np.concatenate([[0.0], s_face]); s_hi = np.concatenate([s_face, [1.0]])
    dsm = s_hi - s_lo; ds_face = np.diff(s)
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    c_prev = C0
    for n in range(nmax):
        tn = (n + 1) * dt
        R = R_of(tn + 0.5 * dt); R2 = R * R
        Tn = T.copy(); Cn = C.copy()
        for it in range(60):
            Tp = Tn.copy(); Cp = Cn.copy()
            vv = vfun(Cp)
            cum = np.concatenate([[0.0], np.cumsum(0.5 * (vv[:-1] + vv[1:]) * ds_face)])
            I = cum[-1]; J = cum / I
            Jf = 0.5 * (J[:-1] + J[1:]); vf = 0.5 * (vv[:-1] + vv[1:])
            kf = kcond(0.5 * (Cp[:-1] + Cp[1:])); Df = Dfun(0.5 * (Cp[:-1] + Cp[1:]), 0.5 * (Tn[:-1] + Tn[1:]))
            coef = 4.0 * I * I / R2
            gC = coef * Jf * Df / (vf * vf * ds_face)
            gT = coef * Jf * kf / (vf * ds_face)
            gC_out = 2.0 * I * hm / (R * vv[-1]); gT_out = 2.0 * I * h / R
            capT = (1.0 + Cp) * cp(Cp)
            # 热
            a = np.zeros(N + 1); b = np.zeros(N + 1); c = np.zeros(N + 1); d = np.zeros(N + 1)
            lout = gT_out * dt / (dsm[N] * capT[N]); l0 = gT[0] * dt / (dsm[0] * capT[0])
            b[0] = 1 + l0; c[0] = -l0; d[0] = T[0]
            for i in range(1, N):
                lL = gT[i - 1] * dt / (dsm[i] * capT[i]); lR = gT[i] * dt / (dsm[i] * capT[i])
                a[i] = -lL; c[i] = -lR; b[i] = 1 + lL + lR; d[i] = T[i]
            lL = gT[N - 1] * dt / (dsm[N] * capT[N]); a[N] = -lL; b[N] = 1 + lL + lout; d[N] = T[N] + lout * Tbc(tn)
            Tn = thomas(a, b, c, d)
            # 质
            a = np.zeros(N + 1); b = np.zeros(N + 1); c = np.zeros(N + 1); d = np.zeros(N + 1)
            lout = gC_out * dt / dsm[N]; l0 = gC[0] * dt / dsm[0]
            b[0] = 1 + l0; c[0] = -l0; d[0] = C[0]
            for i in range(1, N):
                lL = gC[i - 1] * dt / dsm[i]; lR = gC[i] * dt / dsm[i]
                a[i] = -lL; c[i] = -lR; b[i] = 1 + lL + lR; d[i] = C[i]
            lL = gC[N - 1] * dt / dsm[N]; a[N] = -lL; b[N] = 1 + lL + lout; d[N] = C[N] + lout * Cbc(tn)
            Cn = thomas(a, b, c, d)
            if np.max(np.abs(Tn - Tp)) < 1e-6 and np.max(np.abs(Cn - Cp)) < 1e-8:
                break
        T = Tn; C = Cn
        if C[0] < C_target:
            return (n + (C_target - c_prev) / (C[0] - c_prev)) * dt / 3600.0
        c_prev = C[0]
    return None

print('路线B(非均匀收缩) 烘干时长收敛性：')
print('网格收敛(dt=60s):')
for N in [100, 200, 400]:
    print(f'  N={N}: {drying_time(N, 60.0):.4f} h')
print('时间步收敛(N=200):')
for dt in [60.0, 30.0]:
    print(f'  dt={dt}: {drying_time(200, dt):.4f} h')
print('完成')
