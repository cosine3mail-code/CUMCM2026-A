# -*- coding: utf-8 -*-
"""
二维轴对称复核：验证「无限长圆柱(一维径向)」假设。
在 r-z 轴对称域 [0,R]×[0,H] 上解扩散方程(常数 D，含端面对流)，
对比二维中心点 C(0,H,t) 与一维解 C(0,t) 的差异。
几何：R=2cm, 半长 H=12.5cm（全长25cm）；端面 z=0 对流、中面 z=H 对称、侧面 r=R 对流。
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import j0, j1
from scipy.optimize import brentq

R = 0.02; H = 0.125
D = 5e-9; hm = 8e-7; C0 = 2.55; Cair = 0.0
t_eval = 3600.0; dt = 60.0
n_steps = int(t_eval / dt)

# ===================== 一维解析解（Bessel 级数，中心点） =====================
Bi = hm * R / D
def bessel_roots(Bi, n_terms):
    g = np.linspace(1e-6, 200.0, 200000)
    f = g * j1(g) - Bi * j0(g)
    roots = []
    for i in range(len(g) - 1):
        if f[i] * f[i + 1] < 0:
            roots.append(brentq(lambda x: x * j1(x) - Bi * j0(x), g[i], g[i + 1]))
            if len(roots) >= n_terms:
                break
    return np.array(roots)

lam = bessel_roots(Bi, 60)
A = 2.0 * j1(lam) / (lam * (j0(lam) ** 2 + j1(lam) ** 2))
C_1d_center = Cair + (C0 - Cair) * np.sum(A * np.exp(-(lam ** 2) * D * t_eval / R ** 2))

# ===================== 二维数值解（隐式 FV，常数 D） =====================
Nr = 40; Nz = 120
dr = R / Nr; dz = H / Nz
r = np.linspace(0, R, Nr + 1)
z = np.linspace(0, H, Nz + 1)
r_f = 0.5 * (r[:-1] + r[1:])
r_lo = np.concatenate([[0.0], r_f])
r_hi = np.concatenate([r_f, [R]])
annulus = np.pi * (r_hi ** 2 - r_lo ** 2)          # 节点环形面积（轴向单位长度）
vol = annulus * dz                                 # 节点体积（与 z 无关，长度 Nr+1）

def idx(i, j):
    return i + j * (Nr + 1)

n = (Nr + 1) * (Nz + 1)
C = np.full((Nr + 1, Nz + 1), C0)

for step in range(n_steps):
    A = lil_matrix((n, n)); b = np.zeros(n)
    for j in range(Nz + 1):
        for i in range(Nr + 1):
            k = idx(i, j)
            diag = vol[i] / dt
            # 径向内部面
            if i < Nr:
                g = 2 * np.pi * r_f[i] * dz * D / dr
                diag += g; A[k, idx(i + 1, j)] -= g
            if i > 0:
                g = 2 * np.pi * r_f[i - 1] * dz * D / dr
                diag += g; A[k, idx(i - 1, j)] -= g
            # 轴向内部面
            if j < Nz:
                g = annulus[i] * D / dz
                diag += g; A[k, idx(i, j + 1)] -= g
            if j > 0:
                g = annulus[i] * D / dz
                diag += g; A[k, idx(i, j - 1)] -= g
            # 边界对流
            if i == Nr:      # 侧面 r=R
                g = 2 * np.pi * R * dz * hm
                diag += g; b[k] += g * Cair
            if j == 0:       # 端面 z=0
                g = annulus[i] * hm
                diag += g; b[k] += g * Cair
            # i=0 (轴)、j=Nz (中面) 为对称，无通量
            A[k, k] = diag
            b[k] += vol[i] / dt * C[i, j]
    C = spsolve(csr_matrix(A), b).reshape(Nz + 1, Nr + 1).T

C_2d_center = C[0, -1]       # (r=0, z=H) 中面中心
C_2d_end_center = C[0, 0]    # (r=0, z=0) 端面中心

print("===== 二维轴对称复核（常数 D，验证一维假设） =====")
print(f"几何：R={R*100:.1f}cm，半长 H={H*100:.1f}cm（长径比 {H/R:.1f}）")
print(f"t={t_eval}s，一维中心 C(0) = {C_1d_center:.6f} kg/kg")
print(f"二维中面中心 C(0,H) = {C_2d_center:.6f} kg/kg")
print(f"二维端面中心 C(0,0) = {C_2d_end_center:.6f} kg/kg")
rel = (C_2d_center - C_1d_center) / C_1d_center
print(f"中面中心相对偏差 = {rel*100:.3f} %（一维假设误差）")
print(f"端面影响：端面中心比中面中心低 {(C_2d_end_center-C_2d_center):.5f} kg/kg（局部化）")
# 轴向影响范围：找 z 方向 C 从端面衰减到中面 95% 的位置
mid = C[0, -1]; thr = C[0, 0] + 0.95 * (mid - C[0, 0])
jz = np.argmax(C[0, :] >= thr)
print(f"端面影响仅波及 z ≤ {z[jz]*100:.2f} cm（半长 {H*100:.1f} cm 内）")
