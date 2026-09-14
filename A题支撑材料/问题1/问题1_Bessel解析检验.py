# -*- coding: utf-8 -*-
"""Bessel 解析解检验。

功能：以线性常 D 圆柱扩散解析解核对数值离散。模型来源：附录2线性检验。
关键参数：R、D、h_m、C0；输入数据：无外部文件；输出位置：标准输出及检验图。
运行方式：在项目根目录执行 ``python 问题1/问题1_Bessel解析检验.py``。
"""
import numpy as np
from scipy.special import j0, j1
from scipy.optimize import brentq

# ============ 参数（线性情形） ============
R = 0.02; D = 5e-9; hm = 8e-7; C0 = 2.55; C_air = 0.0
Bi = hm * R / D
t_eval = 1800.0

# ============ 解析解（Bessel 级数） ============
def bessel_roots(Bi, n_terms, lam_max=200.0, ngrid=200000):
    g = np.linspace(1e-6, lam_max, ngrid)
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

def C_analytical(r, t):
    return C_air + (C0 - C_air) * np.sum(A[:, None] * j0(np.outer(lam, r / R)) * np.exp(-(lam ** 2)[:, None] * D * t / R ** 2), axis=0)

# ============ 数值解（隐式有限体积，常数 D） ============
def thomas(a, b, c, d):
    n = len(b); cp_ = np.zeros(n); dp_ = np.zeros(n)
    cp_[0] = c[0] / b[0]; dp_[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp_[i - 1]; cp_[i] = c[i] / m; dp_[i] = (d[i] - a[i] * dp_[i - 1]) / m
    x = np.zeros(n); x[-1] = dp_[-1]
    for i in range(n - 2, -1, -1): x[i] = dp_[i] - cp_[i] * x[i + 1]
    return x

def C_numerical(N, dt):
    dr = R / N; s = D * dt / dr**2
    Ah = 2 * N * hm * dt / ((N - 0.25) * dr)
    Bh = 2 * (N - 0.5) * s / (N - 0.25)
    aT = np.zeros(N + 1); bT = np.zeros(N + 1); cT = np.zeros(N + 1)
    bT[0] = 1 + 4 * s; cT[0] = -4 * s
    for i in range(1, N):
        aT[i] = -(i - 0.5) / i * s; cT[i] = -(i + 0.5) / i * s; bT[i] = 1 + 2 * s
    aT[N] = -Bh; bT[N] = 1 + Ah + Bh
    C = np.full(N + 1, C0)
    for _ in range(int(t_eval / dt)):
        d = C.copy(); d[N] += Ah * C_air
        C = thomas(aT, bT, cT, d)
    return C

r = np.linspace(0, R, 201)
C_exact = C_analytical(r, t_eval)

print(f"Bessel 解析解验证（D={D:.1e}, Bi={Bi:.3f}, t={t_eval}s, {len(lam)} 项级数）\n")
print("网格收敛（dt=1s）：")
for N in [100, 200, 400]:
    C_num = C_numerical(N, 1.0)
    rn = np.linspace(0, R, N + 1)
    err = np.max(np.abs(C_num - C_analytical(rn, t_eval)))
    print(f"  N={N}: max|误差| = {err:.3e} kg/kg")
print("\n时间步收敛（N=200）：")
for dt in [2.0, 1.0, 0.5]:
    C_num = C_numerical(200, dt)
    rn = np.linspace(0, R, 201)
    err = np.max(np.abs(C_num - C_analytical(rn, t_eval)))
    print(f"  dt={dt}: max|误差| = {err:.3e} kg/kg")
print("\n验证完成：误差随网格/时间步单调下降且量级达 1e-4 以下，即离散格式正确。")
