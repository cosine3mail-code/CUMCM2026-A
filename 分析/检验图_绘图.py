# -*- coding: utf-8 -*-
"""问题2、3 模型检验展示图：收敛性 + 质量守恒 + 径向分布 + 敏感性"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

R = 0.02; h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55

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

def solve(N, dt, t_end, C_target=None, t_snap=None):
    """求解并记录：总含水量序列 + 指定时刻的 T/C 剖面"""
    dr = R / N; nmax = int(round(t_end / dt))
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    r = np.arange(N + 1) * dr
    m0 = np.trapezoid(C * r, r)
    t_hist = [0.0]; m_hist = [1.0]
    snap = {} if t_snap is None else {t: None for t in t_snap}
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
        t_hist.append(tn); m_hist.append(np.trapezoid(C * r, r) / m0)
        if snap is not None:
            for ts in snap:
                if snap[ts] is None and abs(tn - ts) < dt:
                    snap[ts] = (T.copy(), C.copy())
        if C_target is not None and C[0] < C_target:
            break
    return np.array(t_hist), np.array(m_hist), r, snap

# ===================== 问题2 =====================
t2, m2, r2, s2 = solve(200, 1.0, 10800.0, t_snap=[1800.0, 5400.0, 10800.0])

# ===================== 问题3 =====================
t3, m3, r3, s3 = solve(200, 60.0, 500000.0, C_target=0.15, t_snap=[21600.0, 86400.0, 172800.0, 207000.0])

# ===================== 问题2 检验图 =====================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
# (a) 网格收敛
ax = axes[0, 0]
N = [100, 200, 400]
Cs = [1.008121, 1.008130, 1.008133]
Cc = [1.766194, 1.766195, 1.766195]
ax.plot(N, Cs, 'o-', color='#d62728', label='表面 C')
ax.set_xlabel('网格数 N'); ax.set_ylabel('表面水分 C / (kg/kg)')
ax.set_title('(a) 网格收敛（3h 表面水分）')
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(N, Cc, 's--', color='#1f77b4', label='中心 C')
ax2.set_ylabel('中心水分 C / (kg/kg)')
ax2.ticklabel_format(useOffset=False)
# (b) 时间收敛
ax = axes[0, 1]
dt = [1.0, 0.5, 0.25]
Csc = [1.008130, 1.008124, 1.008121]
Ccc = [1.766195, 1.766184, 1.766179]
ax.plot(dt, Csc, 'o-', color='#d62728', label='表面 C')
ax.set_xlabel('时间步 dt / s'); ax.set_ylabel('表面水分 C / (kg/kg)')
ax.set_title('(b) 时间收敛（3h 表面水分）')
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(dt, Ccc, 's--', color='#1f77b4', label='中心 C')
ax2.set_ylabel('中心水分 C / (kg/kg)')
ax2.ticklabel_format(useOffset=False)
# (c) 质量守恒
ax = axes[1, 0]
ax.plot(t2 / 3600.0, m2, 'b-', lw=1.6)
ax.set_xlabel('时间 / h'); ax.set_ylabel('总含水量比（末/初）')
ax.set_title('(c) 质量守恒（总含水量单调下降）')
ax.grid(alpha=0.3)
# (d) 径向分布
ax = axes[1, 1]
r_cm = r2 * 100
for ts, lab in [(1800.0, '0.5 h'), (5400.0, '1.5 h'), (10800.0, '3 h')]:
    T, C = s2[ts]
    ax.plot(r_cm, C, label=lab)
ax.set_xlabel('到中心距离 r / cm'); ax.set_ylabel('水分浓度 C / (kg/kg)')
ax.set_title('(d) 径向水分分布（外干内湿）')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('图：问题2 模型检验', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('分析/问题2_模型检验图.pdf')

# ===================== 问题3 检验图 =====================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
# (a) 网格收敛
ax = axes[0, 0]
N = [100, 200, 400]
td = [57.7060, 57.6231, 57.5923]
ax.plot(N, td, 'o-', color='#d62728')
ax.set_xlabel('网格数 N'); ax.set_ylabel('烘干时长 / h')
ax.set_title('(a) 网格收敛')
ax.grid(alpha=0.3)
# (b) 时间收敛
ax = axes[0, 1]
dt = [60.0, 30.0, 15.0]
td = [57.6231, 57.5970, 57.5839]
ax.plot(dt, td, 'o-', color='#d62728')
ax.set_xlabel('时间步 dt / s'); ax.set_ylabel('烘干时长 / h')
ax.set_title('(b) 时间收敛')
ax.grid(alpha=0.3)
# (c) 质量守恒
ax = axes[1, 0]
ax.plot(t3 / 3600.0, m3, 'b-', lw=1.4)
ax.set_xlabel('时间 / h'); ax.set_ylabel('总含水量比（末/初）')
ax.set_title('(c) 质量守恒（总含水量单调下降）')
ax.grid(alpha=0.3)
# (d) 敏感性
ax = axes[1, 1]
Tinf = [49.76, 49.94, 49.97, 50.165, 50.25]
tdT = [58.0096, 57.6781, 57.6231, 57.2675, 57.1134]
ax.plot(Tinf, tdT, 'o-', color='#d62728')
ax.axvline(49.97, color='0.5', ls='--', lw=0.9)
ax.set_xlabel('恒温段温度 $T_\\infty$ / ℃'); ax.set_ylabel('烘干时长 / h')
ax.set_title('(d) 恒温段温度敏感性（虚线=平台均值 49.97℃）')
ax.grid(alpha=0.3)
fig.suptitle('图：问题3 模型检验', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('分析/问题3_模型检验图.pdf')

print('已生成 分析/问题2_模型检验图.pdf 与 分析/问题3_模型检验图.pdf')
