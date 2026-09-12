# -*- coding: utf-8 -*-
"""问题2、3 模型检验展示图（改进版）：对数误差收敛图 + 质量守恒 + 表面C随时间 + 敏感性"""
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

def solve(N, dt, t_end, C_target=None, record=False):
    """返回 (表面C, 中心C) 或 (烘干时长h,)，可选记录时间序列"""
    dr = R / N; nmax = int(round(t_end / dt))
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    r = np.arange(N + 1) * dr; m0 = np.trapezoid(C * r, r)
    t_rec, m_rec, Cs_rec = [], [], []
    c_prev = C0
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
        if record:
            t_rec.append(tn); m_rec.append(np.trapezoid(C * r, r) / m0); Cs_rec.append(C[N])
        if C_target is not None and C[0] < C_target:
            td = (n + (C_target - c_prev) / (C[0] - c_prev)) * dt / 3600.0
            if record:
                return td, np.array(t_rec), np.array(m_rec), np.array(Cs_rec)
            return td
        c_prev = C[0]
    if record:
        return (C[N], C[0]), np.array(t_rec), np.array(m_rec), np.array(Cs_rec)
    return (C[N], C[0])

# ===================== 收敛数据 =====================
N2 = [50, 100, 200, 400]
Cs2 = [solve(N, 1.0, 10800.0)[0] for N in N2]
dt2 = [2.0, 1.0, 0.5, 0.25]
Csd2 = [solve(200, d, 10800.0)[0] for d in dt2]
N3 = [50, 100, 200, 400]
td3 = [solve(N, 60.0, 500000.0, C_target=0.15) for N in N3]
dt3 = [120.0, 60.0, 30.0, 15.0]
tdd3 = [solve(200, d, 500000.0, C_target=0.15) for d in dt3]

err_Cs2 = [abs(c - Cs2[-1]) for c in Cs2[:-1]]
err_Csd2 = [abs(c - Csd2[-1]) for c in Csd2[:-1]]
err_td3 = [abs(t - td3[-1]) for t in td3[:-1]]
err_tdd3 = [abs(t - tdd3[-1]) for t in tdd3[:-1]]

# 质量守恒 + 表面C随时间（问题2）
(_, _), t2, m2, Cs2t = solve(200, 1.0, 10800.0, record=True)
# 质量守恒（问题3）
td3_full, t3, m3, _ = solve(200, 60.0, 500000.0, C_target=0.15, record=True)

# ===================== 问题2 检验图 =====================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
# (a) 网格收敛误差
ax = axes[0, 0]
ax.loglog(N2[:-1], err_Cs2, 'o-', color='#d62728', label='数值误差')
ax.loglog(N2[:-1], [err_Cs2[0] * (N2[0] / n) ** 2 for n in N2[:-1]], '--', color='0.5', label='二阶参考')
ax.set_xlabel('网格数 N'); ax.set_ylabel('表面 C 误差 / (kg/kg)')
ax.set_title('(a) 网格收敛（相对 N=400，二阶）')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
# (b) 时间收敛误差
ax = axes[0, 1]
ax.loglog(dt2[:-1], err_Csd2, 'o-', color='#d62728', label='数值误差')
ax.loglog(dt2[:-1], [err_Csd2[0] * (dt2[0] / d) for d in dt2[:-1]], '--', color='0.5', label='一阶参考')
ax.set_xlabel('时间步 dt / s'); ax.set_ylabel('表面 C 误差 / (kg/kg)')
ax.set_title('(b) 时间收敛（相对 dt=0.25，一阶）')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
# (c) 质量守恒
ax = axes[1, 0]
ax.plot(t2 / 3600.0, m2, 'b-', lw=1.6)
ax.set_xlabel('时间 / h'); ax.set_ylabel('总含水量比（末/初）')
ax.set_title('(c) 质量守恒（总含水量单调下降）')
ax.grid(alpha=0.3)
# (d) 表面/中心 C 随时间
ax = axes[1, 1]
ax.plot(t2 / 3600.0, Cs2t, 'r-', lw=1.5, label='表面 C')
ax.set_xlabel('时间 / h'); ax.set_ylabel('水分浓度 C / (kg/kg)')
ax.set_title('(d) 表面水分随时间（快速干燥）')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('图：问题2 模型检验', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('分析/问题2_模型检验图.pdf')

# ===================== 问题3 检验图 =====================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
# (a) 网格收敛误差
ax = axes[0, 0]
ax.loglog(N3[:-1], err_td3, 'o-', color='#d62728', label='数值误差')
ax.loglog(N3[:-1], [err_td3[0] * (N3[0] / n) ** 2 for n in N3[:-1]], '--', color='0.5', label='二阶参考')
ax.set_xlabel('网格数 N'); ax.set_ylabel('烘干时长误差 / h')
ax.set_title('(a) 网格收敛（相对 N=400，二阶）')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
# (b) 时间收敛误差
ax = axes[0, 1]
ax.loglog(dt3[:-1], err_tdd3, 'o-', color='#d62728', label='数值误差')
ax.loglog(dt3[:-1], [err_tdd3[0] * (dt3[0] / d) for d in dt3[:-1]], '--', color='0.5', label='一阶参考')
ax.set_xlabel('时间步 dt / s'); ax.set_ylabel('烘干时长误差 / h')
ax.set_title('(b) 时间收敛（相对 dt=15，一阶）')
ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
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
ax.set_title('(d) 恒温段温度敏感性（虚线=平台均值）')
ax.grid(alpha=0.3)
fig.suptitle('图：问题3 模型检验', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('分析/问题3_模型检验图.pdf')

print('已生成 分析/问题2_模型检验图.pdf 与 分析/问题3_模型检验图.pdf')
