# -*- coding: utf-8 -*-
"""
问题二联合 Picard 迭代收敛性检验

功能：在 0.5 h、1.5 h 和 3.0 h 记录联合 Picard 迭代中
温度场与水分浓度场的最大迭代残差，并输出论文插图：
图9_联合Picard迭代收敛性检验.pdf

使用方法：将本文件置于“问题2”文件夹内运行。
"""

import os
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator


# ===================== 路径与计算参数 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(SCRIPT_DIR, "结果")
ATT1 = os.path.join(ROOT, "附件", "附件1.xlsx")
os.makedirs(RESULT_DIR, exist_ok=True)

R = 0.02
h = 25.0
hm = 8e-7
T0 = 28.0
C0 = 2.55
N = 200
dr = R / N
dt = 1.0
t_end = 10800.0
n_steps = int(round(t_end / dt))

# 迭代记录时刻（s）
RECORD_TIMES = [1800, 5400, 10800]


# ===================== 附录3物性参数 =====================
def rho(C):
    return 650.0 + 128.0 * C


def cp(C):
    return 1450.0 + 2736.0 * C / (C + 1.0)


def kcond(C):
    return 0.21 + 0.38 * C / (C + 1.0)


def Dfun(C, T_C):
    """水分扩散系数，温度 T_C 的单位为摄氏度。"""
    T_K = T_C + 273.15
    return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T_K)


# ===================== 环境边界 =====================
def read_attachment1(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    t = np.array([row[0] for row in rows], dtype=float)
    T = np.array([row[1] for row in rows], dtype=float)
    C = np.array([row[2] for row in rows], dtype=float)
    return t, T, C


t_air, T_air, C_air = read_attachment1(ATT1)
T_air_interp = PchipInterpolator(t_air, T_air)
C_air_interp = PchipInterpolator(t_air, C_air)


def T_bc(t):
    return T_air_interp(t) if t <= 14400.0 else 49.97


def C_bc(t):
    return C_air_interp(t) if t <= 14400.0 else 0.0499


# ===================== Thomas追赶法 =====================
def thomas(a, b, c, d):
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()
    n = len(b)

    c_star = np.zeros(n)
    d_star = np.zeros(n)
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i] * c_star[i - 1]
        c_star[i] = c[i] / denominator
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denominator

    x = np.zeros(n)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    return x


# ===================== 温度场与水分浓度场求解 =====================
def solve_heat(rhocp, k_face, T_old, t_new):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    lam0 = k_face[0] * dt / (rhocp[0] * dr**2)
    b[0] = 1.0 + 4.0 * lam0
    c[0] = -4.0 * lam0
    d[0] = T_old[0]

    for i in range(1, N):
        lam_left = k_face[i - 1] * dt / (rhocp[i] * dr**2)
        lam_right = k_face[i] * dt / (rhocp[i] * dr**2)
        a[i] = -(i - 0.5) / i * lam_left
        c[i] = -(i + 0.5) / i * lam_right
        b[i] = 1.0 + (i - 0.5) / i * lam_left + (i + 0.5) / i * lam_right
        d[i] = T_old[i]

    A_h = 2.0 * N * h * dt / ((N - 0.25) * rhocp[N] * dr)
    lam_N = k_face[N - 1] * dt / (rhocp[N] * dr**2)
    B_h = 2.0 * (N - 0.5) * lam_N / (N - 0.25)
    a[N] = -B_h
    b[N] = 1.0 + A_h + B_h
    d[N] = T_old[N] + A_h * T_bc(t_new)

    return thomas(a, b, c, d)


def solve_mass(D_face, C_old, t_new):
    lam = D_face * dt / dr**2
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    b[0] = 1.0 + 4.0 * lam[0]
    c[0] = -4.0 * lam[0]
    d[0] = C_old[0]

    for i in range(1, N):
        a[i] = -(i - 0.5) / i * lam[i - 1]
        c[i] = -(i + 0.5) / i * lam[i]
        b[i] = 1.0 + (i - 0.5) / i * lam[i - 1] + (i + 0.5) / i * lam[i]
        d[i] = C_old[i]

    A_m = 2.0 * N * hm * dt / ((N - 0.25) * dr)
    B_m = 2.0 * (N - 0.5) * lam[N - 1] / (N - 0.25)
    a[N] = -B_m
    b[N] = 1.0 + A_m + B_m
    d[N] = C_old[N] + A_m * C_bc(t_new)

    return thomas(a, b, c, d)


# ===================== 联合Picard迭代 =====================
T = np.full(N + 1, T0)
C = np.full(N + 1, C0)
picard_history = {}

for n in range(n_steps):
    t_new = (n + 1) * dt
    T_new = T.copy()
    C_new = C.copy()
    residuals = []

    for q in range(50):
        T_previous = T_new.copy()
        C_previous = C_new.copy()

        rhocp = rho(C_previous) * cp(C_previous)
        k_face = kcond(0.5 * (C_previous[:-1] + C_previous[1:]))
        T_new = solve_heat(rhocp, k_face, T, t_new)

        C_face = 0.5 * (C_previous[:-1] + C_previous[1:])
        T_face = 0.5 * (T_new[:-1] + T_new[1:])
        D_face = Dfun(C_face, T_face)
        C_new = solve_mass(D_face, C, t_new)

        residual_T = np.max(np.abs(T_new - T_previous))
        residual_C = np.max(np.abs(C_new - C_previous))
        residuals.append([residual_T, residual_C])

        if residual_T < 1e-6 and residual_C < 1e-8:
            break

    T = T_new
    C = C_new

    if int(round(t_new)) in RECORD_TIMES:
        picard_history[int(round(t_new))] = np.array(residuals)


# ===================== 绘制收敛性检验图 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
colors = ["#2F5597", "#C55A11", "#548235"]

for color, t_show in zip(colors, RECORD_TIMES):
    history = picard_history[t_show]
    q = np.arange(1, len(history) + 1)
    label = f"{t_show / 3600:.1f} h"

    axes[0].semilogy(
        q, np.maximum(history[:, 0], 1e-16),
        marker="o", markersize=5, linewidth=1.5,
        color=color, label=label
    )
    axes[1].semilogy(
        q, np.maximum(history[:, 1], 1e-16),
        marker="s", markersize=5, linewidth=1.5,
        color=color, label=label
    )

axes[0].axhline(1e-6, color="gray", linestyle="--", linewidth=1.0)
axes[1].axhline(1e-8, color="gray", linestyle="--", linewidth=1.0)

axes[0].set_xlabel("Picard 迭代次数")
axes[0].set_ylabel(r"温度迭代残差（$^\circ$C）")
axes[0].set_title("(a) 温度场迭代残差")

axes[1].set_xlabel("Picard 迭代次数")
axes[1].set_ylabel("水分浓度迭代残差（kg/kg）")
axes[1].set_title("(b) 水分浓度场迭代残差")

for ax in axes:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(RESULT_DIR, "图9_联合Picard迭代收敛性检验.pdf")
plt.savefig(output_path, bbox_inches="tight")
plt.show()

print("联合 Picard 迭代收敛性检验图已保存至：")
print(output_path)

for t_show in RECORD_TIMES:
    history = picard_history[t_show]
    print(
        f"t={t_show / 3600:.1f} h："
        f"迭代 {len(history)} 次，"
        f"最终温度残差 {history[-1, 0]:.3e}，"
        f"最终水分残差 {history[-1, 1]:.3e}"
    )
