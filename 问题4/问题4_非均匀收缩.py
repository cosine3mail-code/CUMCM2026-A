# -*- coding: utf-8 -*-
"""
问题4 路线B：非均匀收缩模型（干物质质量分数坐标 + 局部比容）
  v(C) = (1+C)/rho(C) 为局部比容（单位干物质体积）；
  J(s) = ∫0^s v dξ / ∫0^1 v dξ，物理半径 r = R(t)*sqrt(J(s))；
  水分: dC/dt = (4 I^2/R^2) d/ds( (J D / v^2) dC/ds )，I=∫0^1 v dξ；
  热:   capT dT/dt，capT=(1+C)cp(C)（质量模式热容，单位干物质）。
附件2 的 R(t) 作整体标定；附录4 定局部收缩分布。
"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(ROOT, '问题4', '结果')
os.makedirs(RESULT_DIR, exist_ok=True)

h = 25.0; hm = 8e-7; T0 = 28.0; C0 = 2.55; C_target = 0.15
N = 200; dt = 60.0; nmax = int(500000.0 / dt)

def rho(C):   return 760.0 + 90.0 * C
def cp(C):    return 1850.0 + 2150.0 * C / (C + 1.0)
def kcond(C): return 0.12 + 0.20 * C / (C + 1.0)
def Dfun(C, TC): return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / (TC + 273.15))
def vfun(C):  return (1.0 + C) / rho(C)

def read(path, ncol):
    wb = openpyxl.load_workbook(path, data_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    return tuple(np.array([r[j] for r in rows], float) for j in range(ncol))

t_air, T_air, C_air = read(os.path.join(ROOT, '附件', '附件1.xlsx'), 3)
Tair = PchipInterpolator(t_air, T_air); Cair = PchipInterpolator(t_air, C_air)
t_rad, R_rad = read(os.path.join(ROOT, '附件', '附件2.xlsx'), 2)
R_rad = R_rad / 100.0
Rp = PchipInterpolator(t_rad, R_rad); R_min = float(R_rad[-1])

def Tbc(t): return Tair(t) if t <= 14400.0 else 49.97   # 平台均值（附件1 末2h平均）
def Cbc(t): return Cair(t) if t <= 14400.0 else 0.0499
def R_of(t): return R_min if t >= t_rad[-1] else float(Rp(t))

def thomas(a, b, c, d):
    a = a.copy(); b = b.copy(); c = c.copy(); d = d.copy(); n = len(b)
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

# 干物质质量分数网格 s=(i/N)^2，使物理半径 r≈R*(i/N) 近似均匀
s = (np.arange(N + 1) / N) ** 2
s_face = 0.5 * (s[:-1] + s[1:])
s_lo = np.concatenate([[0.0], s_face])
s_hi = np.concatenate([s_face, [1.0]])
dsm = s_hi - s_lo                      # 控制体干物质质量分数（长度 N+1）
ds_face = np.diff(s)                   # 相邻节点间距（长度 N）

def build_tridiag(Y_old, Cap, g, g_out, Yinf):
    """Cap_i (Y_i-Y_old_i)/dt = g_{i-1}(Y_{i-1}-Y_i) - g_i(Y_i-Y_{i+1})，表面 -g_out(Y_N-Yinf)"""
    a = np.zeros(N + 1); b = np.zeros(N + 1); c = np.zeros(N + 1); d = np.zeros(N + 1)
    lam_out = g_out * dt / Cap[N]
    l0 = g[0] * dt / Cap[0]
    b[0] = 1 + l0; c[0] = -l0; d[0] = Y_old[0]
    for i in range(1, N):
        lL = g[i - 1] * dt / Cap[i]
        lR = g[i] * dt / Cap[i]
        a[i] = -lL; c[i] = -lR
        b[i] = 1 + lL + lR; d[i] = Y_old[i]
    lL = g[N - 1] * dt / Cap[N]
    a[N] = -lL; b[N] = 1 + lL + lam_out
    c[N] = 0.0; d[N] = Y_old[N] + lam_out * Yinf
    return a, b, c, d

T = np.full(N + 1, T0); C = np.full(N + 1, C0)
c_center_prev = C0
t_dry = None

for n in range(nmax):
    tn = (n + 1) * dt
    R = R_of(tn + 0.5 * dt); R2 = R * R
    T_new = T.copy(); C_new = C.copy()
    for it in range(60):
        Tp = T_new.copy(); Cp = C_new.copy()
        # 局部比容与映射
        vv = vfun(Cp)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (vv[:-1] + vv[1:]) * ds_face)])
        I = cum[-1]
        J = cum / I
        Jf = 0.5 * (J[:-1] + J[1:]); vf = 0.5 * (vv[:-1] + vv[1:])
        kf = kcond(0.5 * (Cp[:-1] + Cp[1:]))
        Df = Dfun(0.5 * (Cp[:-1] + Cp[1:]), 0.5 * (T_new[:-1] + T_new[1:]))
        coef = 4.0 * I * I / R2
        gC = coef * Jf * Df / (vf * vf * ds_face)
        gT = coef * Jf * kf / (vf * ds_face)
        gC_out = 2.0 * I * hm / (R * vv[-1])
        gT_out = 2.0 * I * h / R
        # 热方程（质量模式热容 capT=(1+C)cp）
        capT = (1.0 + Cp) * cp(Cp)
        aT, bT, cT, dT = build_tridiag(T, dsm * capT, gT, gT_out, Tbc(tn))
        T_new = thomas(aT, bT, cT, dT)
        # 质方程
        aC, bC, cC, dC = build_tridiag(C, dsm, gC, gC_out, Cbc(tn))
        C_new = thomas(aC, bC, cC, dC)
        if np.max(np.abs(T_new - Tp)) < 1e-6 and np.max(np.abs(C_new - Cp)) < 1e-8:
            break
    T = T_new; C = C_new
    if C[0] < C_target:
        t_dry = (n + (C_target - c_center_prev) / (C[0] - c_center_prev)) * dt
        break
    c_center_prev = C[0]

if t_dry is None:
    t_dry = nmax * dt

print("===== 问题4 路线B（非均匀收缩 / 局部比容） =====")
print(f"烘干所需时间：{t_dry:.1f} s = {t_dry/3600.0:.4f} h")
print(f"结束半径 R = {R_of(t_dry)*100:.4f} cm")
print(f"中心 C = {C[0]:.4f}，表面 C = {C[N]:.4f}")
print(f"中心 T = {T[0]:.2f}℃，表面 T = {T[N]:.2f}℃")
