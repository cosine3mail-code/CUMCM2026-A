# -*- coding: utf-8 -*-
"""
问题1 数据处理 + 两种插值方法可视化
1. 用 PCHIP 与 线性 两种插值分别求解 0-1800s 温度/水分场
2. 保存结果数据: result1.xlsx(PCHIP)、表1表2.xlsx、两种插值数组 npz
3. 生成 PDF 图: 图1 线性插值(0-1800s 左右构图)、图2 全范围对比、图3 结果时间序列、图4 结果差异(左右构图)
"""
import os
import numpy as np
import openpyxl
from scipy.interpolate import PchipInterpolator, interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class MathSciFormatter(ticker.ScalarFormatter):
    """科学计数法偏移用 Unicode 上标（×10⁻⁶），避免 mathtext 负号缺字。"""
    def get_offset(self):
        if len(self._locs) == 0:
            return ''
        if self._orderOfMagnitude:
            _sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
            return '×10' + str(self._orderOfMagnitude).translate(_sup)
        if self.offset:
            s = self.format_data(self.offset)
            if self.offset > 0:
                s = '+' + s
            return s
        return ''

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 路径（脚本位于 问题1/，ROOT 为项目根 A题/） =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
RESULT_DIR = os.path.join(ROOT, '问题1', '结果')
FIG_DIR = os.path.join(ROOT, '问题1', '图')
ATT1 = os.path.join(ROOT, '附件', '附件1.xlsx')

# ===================== 参数(附录2) =====================
R    = 0.02; rho = 820.0; cp = 2600.0; k = 0.36
h    = 25.0;  hm  = 8e-7;  D0 = 7e-9;  Da = 0.89
T0   = 28.0;  C0  = 2.55
t_end = 1800.0; dt = 1.0; N = 200; dr = R / N; alpha = k / (rho * cp)


def Dfun(C):
    return D0 * np.exp(-Da / C)


# ===================== 读附件1 =====================
def read_att1(path):
    wb = openpyxl.load_workbook(path, data_only=True); ws = wb.active
    rows = list(ws.iter_rows(values_only=True))[1:]
    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float),
            np.array([r[2] for r in rows], float))

t_air, T_air, C_air = read_att1(ATT1)


def make_interp(kind):
    if kind == 'pchip':
        return PchipInterpolator(t_air, T_air), PchipInterpolator(t_air, C_air)
    else:
        return interp1d(t_air, T_air, kind='linear'), interp1d(t_air, C_air, kind='linear')


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


# ===================== 求解器 =====================
def solve(kind):
    Tair, Cair = make_interp(kind)
    s = alpha * dt / dr**2
    Ah = 2 * N * h * dt / ((N - 0.25) * rho * cp * dr)
    Bh = 2 * (N - 0.5) * s / (N - 0.25)
    aT = np.zeros(N + 1); bT = np.zeros(N + 1); cT = np.zeros(N + 1)
    bT[0] = 1 + 4 * s; cT[0] = -4 * s
    for i in range(1, N):
        aT[i] = -(i - 0.5) / i * s; cT[i] = -(i + 0.5) / i * s; bT[i] = 1 + 2 * s
    aT[N] = -Bh; bT[N] = 1 + Ah + Bh
    Am = 2 * N * hm * dt / ((N - 0.25) * dr)
    T = np.full(N + 1, T0); C = np.full(N + 1, C0)
    idx_full = np.arange(0, N + 1, 10)
    idx_table = [0, 50, 100, 150, 200]
    n_steps = int(round(t_end / dt))
    T_full = np.zeros((n_steps, len(idx_full))); C_full = np.zeros((n_steps, len(idx_full)))
    times_table = [100, 300, 600, 900, 1200, 1500, 1800]
    tm = {t: i for i, t in enumerate(times_table)}
    T_table = np.zeros((7, 5)); C_table = np.zeros((7, 5))
    for n in range(n_steps):
        tn = (n + 1) * dt
        dT = T.copy(); dT[N] += Ah * Tair(tn); T = thomas(aT, bT, cT, dT)
        Cn = C.copy()
        for it in range(50):
            Cp = Cn.copy(); Cf = 0.5 * (Cp[:-1] + Cp[1:]); lam = Dfun(Cf) * dt / dr**2
            aC = np.zeros(N + 1); bC = np.zeros(N + 1); cC = np.zeros(N + 1); dC = np.zeros(N + 1)
            bC[0] = 1 + 4 * lam[0]; cC[0] = -4 * lam[0]; dC[0] = C[0]
            for i in range(1, N):
                aC[i] = -(i - 0.5) / i * lam[i - 1]; cC[i] = -(i + 0.5) / i * lam[i]
                bC[i] = 1 + (i - 0.5) / i * lam[i - 1] + (i + 0.5) / i * lam[i]; dC[i] = C[i]
            Bm = 2 * (N - 0.5) * lam[N - 1] / (N - 0.25)
            aC[N] = -Bm; bC[N] = 1 + Am + Bm; dC[N] = C[N] + Am * Cair(tn)
            Cn = thomas(aC, bC, cC, dC)
            if np.max(np.abs(Cn - Cp)) < 1e-12:
                break
        C = Cn
        T_full[n] = T[idx_full]; C_full[n] = C[idx_full]
        if tn in tm:
            j = tm[tn]; T_table[j] = T[idx_table]; C_table[j] = C[idx_table]
    return dict(T_full=T_full, C_full=C_full, T_table=T_table, C_table=C_table,
                Tair=Tair, Cair=Cair, T=T, C=C)


res_p = solve('pchip')
res_l = solve('linear')

# ===================== 保存结果数据 =====================
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

r_full = np.arange(0, N + 1, 10) * dr          # 21 个径向点 (0..2cm)
r_vals = [round(float(v) * 100, 1) for v in r_full]
times = list(range(1, int(t_end) + 1))


def write_result1(path, T_full, C_full):
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "温度"
    ws.cell(1, 1, "时间\\到药材中心的距离")
    for j, rv in enumerate(r_vals):
        ws.cell(1, 2 + j, rv)
    for i, t in enumerate(times):
        ws.cell(2 + i, 1, t)
        for j in range(len(r_vals)):
            ws.cell(2 + i, 2 + j, round(float(T_full[i, j]), 4)).number_format = '0.0000'
    ws2 = wb.create_sheet("水分浓度")
    ws2.cell(1, 1, "时间\\到药材中心的距离")
    for j, rv in enumerate(r_vals):
        ws2.cell(1, 2 + j, rv)
    for i, t in enumerate(times):
        ws2.cell(2 + i, 1, t)
        for j in range(len(r_vals)):
            ws2.cell(2 + i, 2 + j, round(float(C_full[i, j]), 4)).number_format = '0.0000'
    wb.save(path)


write_result1(os.path.join(RESULT_DIR, 'result1.xlsx'), res_p['T_full'], res_p['C_full'])

# 表1/表2
wb = openpyxl.Workbook()
ws = wb.active; ws.title = "表1温度"
r_table_cm = [0.0, 0.5, 1.0, 1.5, 2.0]
ws.cell(1, 1, "时间/s"); [ws.cell(1, 2 + j, f"{v:g}") for j, v in enumerate(r_table_cm)]
for i, t in enumerate([100, 300, 600, 900, 1200, 1500, 1800]):
    ws.cell(2 + i, 1, t)
    for j in range(5):
        ws.cell(2 + i, 2 + j, round(float(res_p['T_table'][i, j]), 4)).number_format = '0.0000'
ws2 = wb.create_sheet("表2水分")
ws2.cell(1, 1, "时间/s"); [ws2.cell(1, 2 + j, f"{v:g}") for j, v in enumerate(r_table_cm)]
for i, t in enumerate([100, 300, 600, 900, 1200, 1500, 1800]):
    ws2.cell(2 + i, 1, t)
    for j in range(5):
        ws2.cell(2 + i, 2 + j, round(float(res_p['C_table'][i, j]), 4)).number_format = '0.0000'
wb.save(os.path.join(RESULT_DIR, '表1表2.xlsx'))

np.savez_compressed(os.path.join(RESULT_DIR, '问题1_两种插值.npz'),
                    t=times, r_cm=r_vals,
                    T_full_pchip=res_p['T_full'], C_full_pchip=res_p['C_full'],
                    T_full_linear=res_l['T_full'], C_full_linear=res_l['C_full'])

print("结果数据已保存: result1.xlsx / 表1表2.xlsx / 问题1_两种插值.npz")

# ===================== 图1 附件1线性插值(0-1800s, 左右构图, 无图名无图例) =====================
t_fine = np.arange(0, 1801, 1)
mask = t_air <= 1800

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
# 左: 烘房温度
ax = axes[0]
ax.scatter(t_air[mask], T_air[mask], s=20, marker='o', facecolor='none', edgecolor='lightcoral', zorder=6, label='原始数据点')
ax.plot(t_fine, res_l['Tair'](t_fine), color='red', lw=1.5, label='线性插值')
ax.set_ylabel('烘房温度 $T_{air}$ / ℃'); ax.set_xlabel('时间 $t$ / s')
ax.set_title('（一）环境温度')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
# 右: 烘房水分浓度
ax = axes[1]
ax.scatter(t_air[mask], C_air[mask], s=20, marker='o', facecolor='none', edgecolor='blue', zorder=6, label='原始数据点')
ax.plot(t_fine, res_l['Cair'](t_fine), color='blue', lw=1.5, label='线性插值')
ax.set_ylabel('烘房水分浓度 $C_{air}$ / (kg/kg)'); ax.set_xlabel('时间 $t$ / s')
ax.set_title('（二）水分浓度')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '图1_附件1线性插值_0-1800s.pdf'))


# ===================== 图4 两种插值结果差异(左右构图, 无图名) =====================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
dT_s = res_p['T_full'][:, -1] - res_l['T_full'][:, -1]
dT_c = res_p['T_full'][:, 0] - res_l['T_full'][:, 0]
dC_s = res_p['C_full'][:, -1] - res_l['C_full'][:, -1]
dC_c = res_p['C_full'][:, 0] - res_l['C_full'][:, 0]
ax = axes[0]
ax.plot(t, dT_s, 'b-', lw=1.2, label='表面')
ax.plot(t, dT_c, 'g-', lw=1.2, label='中心')
ax.set_ylabel('ΔT = T(PCHIP) - T(线性) / ℃'); ax.set_xlabel('时间 $t$ / s')
ax.set_title('（一）温度差')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(t, dC_s, 'b-', lw=1.2, label='表面')
ax.plot(t, dC_c, 'g-', lw=1.2, label='中心')
ax.set_ylabel('ΔC = C(PCHIP) - C(线性) / (kg/kg)'); ax.set_xlabel('时间 $t$ / s')
ax.set_title('（二）水分浓度差')
_yfmt = MathSciFormatter()
_yfmt.set_scientific(True)
_yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(_yfmt)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
# 偏移文本换用 DejaVu Sans（SimHei 无上标负号/数字字形）
ax.yaxis.get_offset_text().set_fontfamily('DejaVu Sans')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '图4_求解结果差异.pdf'))

# ===================== 汇总 =====================
print('\n两种插值结果差异汇总:')
print(f'  温度  max|ΔT| = {np.abs(dT_s).max():.3e} ℃ (表面)')
print(f'  水分  max|ΔC| = {np.abs(dC_s).max():.3e} kg/kg (表面)')
print(f'  t=1800s 表面温度: PCHIP={res_p["T_full"][-1,-1]:.4f}  线性={res_l["T_full"][-1,-1]:.4f}')
print(f'  t=1800s 表面水分: PCHIP={res_p["C_full"][-1,-1]:.4f}  线性={res_l["C_full"][-1,-1]:.4f}')
print('\nPDF 图已生成于 问题1/图/ 目录')
