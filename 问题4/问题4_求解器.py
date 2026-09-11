# -*- coding: utf-8 -*-
"""
问题4 求解器：考虑尺寸收缩确定烘干时长
模型：材料(Lagrangian)坐标 xi=r/R(t)，仿射(均匀)收缩，一维轴对称
  水分: dC/dt = (1/(R^2 xi)) d/dxi( xi D dC/dxi )   —— 无对流项(干物质守恒)
  热:   rho*cp dT/dt = (1/(R^2 xi)) d/dxi( xi k dT/dxi )
物性：附录4（rho=760+90C, cp=1850+2150C/(C+1), k=0.12+0.20C/(C+1),
      D=4.2e-4*e^(-0.30/C)*e^(-3850/T)）
几何：R(t) 取自附件2（PCHIP 插值）；h=25、hm=8e-7 沿用附录2
判据：C(r=0)<0.15（中心最高，全剖面单调性同问题3已验证）
"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator

# ===================== 路径 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(ROOT, '问题4', '结果')
os.makedirs(RESULT_DIR, exist_ok=True)
ATT1 = os.path.join(ROOT, '附件', '附件1.xlsx')
ATT2 = os.path.join(ROOT, '附件', '附件2.xlsx')

# ===================== 参数 =====================
h    = 25.0
hm   = 8e-7
T0   = 28.0
C0   = 2.55
C_target = 0.15
N    = 200
dxi  = 1.0 / N
dt   = 60.0
t_max = 500000.0
nmax = int(t_max / dt)


# ===================== 附录4 物性 =====================
def rho(C):   return 760.0 + 90.0 * C
def cp(C):    return 1850.0 + 2150.0 * C / (C + 1.0)
def kcond(C): return 0.12 + 0.20 * C / (C + 1.0)
def Dfun(C, T_C):
    return 4.2e-4 * np.exp(-0.30 / C) * np.exp(-3850.0 / (T_C + 273.15))


# ===================== 读附件1（环境）、附件2（半径） =====================
def read_xlsx(path, ncol):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    return tuple(np.array([r[j] for r in rows], float) for j in range(ncol))

t_air, T_air, C_air = read_xlsx(ATT1, 3)
T_air_i = PchipInterpolator(t_air, T_air)
C_air_i = PchipInterpolator(t_air, C_air)

t_rad, R_rad = read_xlsx(ATT2, 2)
R_rad = R_rad / 100.0                      # cm -> m
R_pchip = PchipInterpolator(t_rad, R_rad)
R0 = float(R_rad[0]); R_min = float(R_rad[-1])


def T_bc(t): return T_air_i(t) if t <= 14400.0 else 50.0
def C_bc(t): return C_air_i(t) if t <= 14400.0 else 0.05

def R_of(t):
    if t >= t_rad[-1]:
        return R_min
    return float(R_pchip(t))


# ===================== Thomas =====================
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


# ===================== 时间推进（材料坐标，耦合 Picard） =====================
T = np.full(N + 1, T0)
C = np.full(N + 1, C0)
c_center_prev = C0
t_dry = None
iter_hist = []
C_hist = []

for n in range(nmax):
    tn = (n + 1) * dt
    R = R_of(tn + 0.5 * dt)                 # 半时步半径
    R2 = R * R
    T_new = T.copy(); C_new = C.copy()
    for it in range(50):
        Tp = T_new.copy(); Cp = C_new.copy()
        rhocp = rho(Cp) * cp(Cp)
        k_face = kcond(0.5 * (Cp[:-1] + Cp[1:]))
        # ---- 热方程 ----
        aT = np.zeros(N + 1); bT = np.zeros(N + 1); cT = np.zeros(N + 1); dT = np.zeros(N + 1)
        l0 = k_face[0] * dt * N * N / (rhocp[0] * R2)
        bT[0] = 1 + 4 * l0; cT[0] = -4 * l0; dT[0] = T[0]
        for i in range(1, N):
            lL = k_face[i - 1] * dt * N * N / (rhocp[i] * R2)
            lR = k_face[i] * dt * N * N / (rhocp[i] * R2)
            aT[i] = -(i - 0.5) / i * lL
            cT[i] = -(i + 0.5) / i * lR
            bT[i] = 1 + (i - 0.5) / i * lL + (i + 0.5) / i * lR
            dT[i] = T[i]
        Ah = 2 * N * N * h * dt / ((N - 0.25) * rhocp[N] * R)
        lN = k_face[N - 1] * dt * N * N / (rhocp[N] * R2)
        Bh = 2 * (N - 0.5) * lN / (N - 0.25)
        aT[N] = -Bh; bT[N] = 1 + Ah + Bh; dT[N] = T[N] + Ah * T_bc(tn)
        T_new = thomas(aT, bT, cT, dT)
        # ---- 质方程 ----
        Cf = 0.5 * (Cp[:-1] + Cp[1:]); Tf = 0.5 * (T_new[:-1] + T_new[1:])
        lam = Dfun(Cf, Tf) * dt * N * N / R2
        aC = np.zeros(N + 1); bC = np.zeros(N + 1); cC = np.zeros(N + 1); dC = np.zeros(N + 1)
        bC[0] = 1 + 4 * lam[0]; cC[0] = -4 * lam[0]; dC[0] = C[0]
        for i in range(1, N):
            aC[i] = -(i - 0.5) / i * lam[i - 1]
            cC[i] = -(i + 0.5) / i * lam[i]
            bC[i] = 1 + (i - 0.5) / i * lam[i - 1] + (i + 0.5) / i * lam[i]
            dC[i] = C[i]
        Am = 2 * N * N * hm * dt / ((N - 0.25) * R)
        Bm = 2 * (N - 0.5) * lam[N - 1] / (N - 0.25)
        aC[N] = -Bm; bC[N] = 1 + Am + Bm; dC[N] = C[N] + Am * C_bc(tn)
        C_new = thomas(aC, bC, cC, dC)
        if np.max(np.abs(T_new - Tp)) < 1e-6 and np.max(np.abs(C_new - Cp)) < 1e-8:
            break
    T = T_new; C = C_new
    iter_hist.append(it + 1)
    C_hist.append(C.copy())
    if C[0] < C_target:
        t_dry = (n + (C_target - c_center_prev) / (C[0] - c_center_prev)) * dt
        break
    c_center_prev = C[0]

if t_dry is None:
    print("警告：未达标")
    t_dry = nmax * dt

print("===== 问题4 结果（仿射收缩 / 材料坐标，附录4） =====")
print(f"烘干所需时间：{t_dry:.1f} s = {t_dry/3600.0:.4f} h")
print(f"结束时刻半径 R = {R_of(t_dry)*100:.4f} cm（初始 2.0 → 终态 {R_min*100:.4f} cm）")
print(f"结束时刻中心水分 C(r=0) = {C[0]:.4f}，表面 C(R) = {C[N]:.4f}")
print(f"结束时刻温度：中心 {T[0]:.2f}℃ → 表面 {T[N]:.2f}℃")
print(f"Picard 迭代：平均 {np.mean(iter_hist):.1f} 次/步，最大 {np.max(iter_hist)} 次")

# ===================== 输出表6 与 result4.xlsx（移动边界） =====================
n_done = len(C_hist)
xi_grid = np.arange(N + 1) / N
r_fixed = np.linspace(0.0, 0.02, 21)              # 0..2cm 共21点
times = [int((i + 1) * dt) for i in range(n_done)]

def C_at(rr, Cc, R):
    if rr > R + 1e-12:
        return None
    return float(np.interp(rr / R, xi_grid, Cc))

# result4.xlsx（固定物理距离列 + 药材表面列，材料外留空）
r_vals = [round(float(v) * 100, 1) for v in r_fixed]
wb = openpyxl.Workbook(); ws = wb.active; ws.title = "水分浓度"
ws.cell(1, 1, "时间\\到药材中心的距离")
for j, rv in enumerate(r_vals):
    ws.cell(1, 2 + j, rv)
ws.cell(1, 2 + len(r_vals), "药材表面")
for i, t in enumerate(times):
    R = R_of(t + 0.5 * dt); Cc = C_hist[i]
    ws.cell(2 + i, 1, t)
    for j, rr in enumerate(r_fixed):
        v = C_at(rr, Cc, R)
        if v is not None:
            ws.cell(2 + i, 2 + j, round(v, 4)).number_format = '0.0000'
    ws.cell(2 + i, 2 + len(r_vals), round(float(Cc[-1]), 4)).number_format = '0.0000'
wb.save(os.path.join(RESULT_DIR, 'result4.xlsx'))

# 表6（每6h，固定距离 0/0.5/1/1.5 cm + 药材表面）
r_tab = [0.0, 0.005, 0.01, 0.015]
print("\n表6  药材烘干过程的水分浓度（kg/kg）")
print("时间/h | " + " | ".join([f"{v*100:g}cm" for v in r_tab] + ["表面"]))
print("-" * 60)
for th in range(6, int(t_dry // 3600) + 1, 6):
    i = int(th * 3600 / dt) - 1
    if i >= n_done:
        break
    R = R_of(th * 3600); Cc = C_hist[i]
    row = []
    for rr in r_tab:
        v = C_at(rr, Cc, R)
        row.append("  --  " if v is None else f"{v:.4f}")
    row.append(f"{Cc[-1]:.4f}")
    print(f"{th:>6d} | " + " | ".join(row))

print(f"\n已写出 问题4/结果/result4.xlsx（{n_done} 行）")
