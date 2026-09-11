# -*- coding: utf-8 -*-
"""问题3 能量审计（潜热反证）+ 恒温段敏感性图"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 能量审计（潜热反证） =====================
R = 0.02; L = 2.4e6          # 水的汽化潜热 J/kg
C0 = 2.55; C_end = 0.15       # 干基含水率初/末
rho0 = 650 + 128 * C0         # 附录3 初始密度 kg/m^3
rho_s = rho0 / (1 + C0)       # 初始干物质密度 kg/m^3（单位体积干物质）
V = np.pi * R ** 2            # 单位长度体积 m^2
m_s = rho_s * V               # 单位长度干物质质量 kg/m
m_w = m_s * (C0 - C_end)      # 单位长度去除的水质量 kg/m
Q_latent = m_w * L            # 蒸发潜热需求 J/m

cp_mean = 2000.0              # 平均比热（干~湿插值）
m_tot = m_s * (1 + C0)        # 单位长度总质量 kg/m
Q_sens = m_tot * cp_mean * 22.0   # 显热(28→50℃) J/m

print("===== 能量审计（潜热反证） =====")
print(f"干物质密度 rho_s = {rho_s:.1f} kg/m^3，单位长度干物质 {m_s:.4f} kg/m")
print(f"去除水分 {m_w:.4f} kg/m（干基 {C0}→{C_end}）")
print(f"蒸发潜热需求 Q_latent = {Q_latent:.3e} J/m")
print(f"显热需求(升温)   Q_sens   = {Q_sens:.3e} J/m")
print(f"比值 Q_latent / Q_sens = {Q_latent/Q_sens:.1f} 倍")
print("结论：蒸发潜热需求约为显热的几十倍；题给 h=25 只够供给显热升温，")
print("不足以供给蒸发潜热 → 题目模型不含潜热，h 是'等效换热系数'（本文口径正确）。\n")

# ===================== 敏感性图 =====================
# 数据来自 敏感性扫描.py
Tinf = np.array([49.76, 49.94, 49.97, 50.00, 50.165, 50.25])
t_T = np.array([58.0111, 57.6796, 57.6246, 57.5697, 57.2690, 57.1149])
Cinf = np.array([0.0489, 0.0495, 0.0499, 0.0500, 0.0503])
t_C = np.array([57.5538, 57.5623, 57.5682, 57.5697, 57.5743])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.plot(Tinf, t_T, 'o-', lw=1.6, color='#d62728', ms=5)
ax.axvline(49.97, color='0.5', ls='--', lw=0.8)
ax.axvline(50.165, color='0.4', ls=':', lw=0.8)
ax.set_xlabel('恒温段温度 $T_\\infty$ / ℃'); ax.set_ylabel('烘干时长 / h')
ax.set_title('（一）$T_\\infty$ 敏感性（$C_\\infty$=0.05）')
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(Cinf, t_C, 's-', lw=1.6, color='#1f77b4', ms=5)
ax.set_xlabel('恒温段水分 $C_\\infty$ / (kg/kg)'); ax.set_ylabel('烘干时长 / h')
ax.set_title('（二）$C_\\infty$ 敏感性（$T_\\infty$=50℃）')
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig('问题3/结果/恒温段敏感性图.pdf')
print("已写出 问题3/结果/恒温段敏感性图.pdf")
