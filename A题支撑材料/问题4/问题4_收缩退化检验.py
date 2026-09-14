# -*- coding: utf-8 -*-
"""问题四模型检验：固定半径退化一致性与收缩条件下的质量平衡。

输入：同目录 ``问题4_求解器.py``；输出：同目录 ``图/``；运行方式：在项目根目录执行本文件。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SOLVER_FILE = Path(__file__).resolve().parent / "问题4_求解器.py"
OUTPUT_DIR = Path(__file__).resolve().parent / "图"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sci_math(value, digits=2):
    """数值 → mathtext 科学计数法（如 3.08×10^{-12}），避免 'e-12' 形式。"""
    if value == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / 10 ** exp
    return rf'{mant:.{digits}f}\times10^{{{exp}}}'


def load_solver_definitions():
    source = SOLVER_FILE.read_text(encoding="utf-8")
    prefix = source.split("# ===================== 时间推进")[0]
    namespace = {"__file__": str(SOLVER_FILE), "__name__": "q4_solver_definitions"}
    exec(compile(prefix, str(SOLVER_FILE), "exec"), namespace)
    return namespace


def thomas_step_material(ns, temperature, concentration, current_time, radius):
    """问题四材料坐标下的联合 Picard 单次推进。"""
    n = ns["N"]
    dt = ns["dt"]
    h = ns["h"]
    hm = ns["hm"]
    rho = ns["rho"]
    cp = ns["cp"]
    kcond = ns["kcond"]
    d_fun = ns["Dfun"]
    t_bc = ns["T_bc"]
    c_bc = ns["C_bc"]
    updated_t = temperature.copy()
    updated_c = concentration.copy()

    for _ in range(50):
        prev_t = updated_t.copy()
        prev_c = updated_c.copy()
        rho_cp = rho(prev_c) * cp(prev_c)
        k_face = kcond(0.5 * (prev_c[:-1] + prev_c[1:]))
        radius_sq = radius**2

        a_t = np.zeros(n + 1)
        b_t = np.zeros(n + 1)
        c_t = np.zeros(n + 1)
        d_t = np.zeros(n + 1)
        l0 = k_face[0] * dt * n**2 / (rho_cp[0] * radius_sq)
        b_t[0] = 1.0 + 4.0 * l0
        c_t[0] = -4.0 * l0
        d_t[0] = temperature[0]
        for i in range(1, n):
            l_left = k_face[i - 1] * dt * n**2 / (rho_cp[i] * radius_sq)
            l_right = k_face[i] * dt * n**2 / (rho_cp[i] * radius_sq)
            a_t[i] = -(i - 0.5) / i * l_left
            c_t[i] = -(i + 0.5) / i * l_right
            b_t[i] = 1.0 - a_t[i] - c_t[i]
            d_t[i] = temperature[i]
        a_h = 2.0 * n**2 * h * dt / ((n - 0.25) * rho_cp[n] * radius)
        l_n = k_face[n - 1] * dt * n**2 / (rho_cp[n] * radius_sq)
        b_h = 2.0 * (n - 0.5) * l_n / (n - 0.25)
        a_t[n] = -b_h
        b_t[n] = 1.0 + a_h + b_h
        d_t[n] = temperature[n] + a_h * t_bc(current_time)
        updated_t = ns["thomas"](a_t, b_t, c_t, d_t)

        c_face = 0.5 * (prev_c[:-1] + prev_c[1:])
        t_face = 0.5 * (updated_t[:-1] + updated_t[1:])
        lam = d_fun(c_face, t_face) * dt * n**2 / radius_sq
        a_c = np.zeros(n + 1)
        b_c = np.zeros(n + 1)
        c_c = np.zeros(n + 1)
        d_c = np.zeros(n + 1)
        b_c[0] = 1.0 + 4.0 * lam[0]
        c_c[0] = -4.0 * lam[0]
        d_c[0] = concentration[0]
        for i in range(1, n):
            a_c[i] = -(i - 0.5) / i * lam[i - 1]
            c_c[i] = -(i + 0.5) / i * lam[i]
            b_c[i] = 1.0 - a_c[i] - c_c[i]
            d_c[i] = concentration[i]
        a_m = 2.0 * n**2 * hm * dt / ((n - 0.25) * radius)
        b_m = 2.0 * (n - 0.5) * lam[n - 1] / (n - 0.25)
        a_c[n] = -b_m
        b_c[n] = 1.0 + a_m + b_m
        d_c[n] = concentration[n] + a_m * c_bc(current_time)
        updated_c = ns["thomas"](a_c, b_c, c_c, d_c)

        if (
            np.max(np.abs(updated_t - prev_t)) < 1.0e-6
            and np.max(np.abs(updated_c - prev_c)) < 1.0e-8
        ):
            break
    return updated_t, updated_c


def thomas_step_fixed(ns, temperature, concentration, current_time, radius):
    """固定物理坐标下的对照模型单次推进。"""
    n = ns["N"]
    dt = ns["dt"]
    dr = radius / n
    h = ns["h"]
    hm = ns["hm"]
    rho = ns["rho"]
    cp = ns["cp"]
    kcond = ns["kcond"]
    d_fun = ns["Dfun"]
    t_bc = ns["T_bc"]
    c_bc = ns["C_bc"]
    updated_t = temperature.copy()
    updated_c = concentration.copy()

    for _ in range(50):
        prev_t = updated_t.copy()
        prev_c = updated_c.copy()
        rho_cp = rho(prev_c) * cp(prev_c)
        k_face = kcond(0.5 * (prev_c[:-1] + prev_c[1:]))

        a_t = np.zeros(n + 1)
        b_t = np.zeros(n + 1)
        c_t = np.zeros(n + 1)
        d_t = np.zeros(n + 1)
        l0 = k_face[0] * dt / (rho_cp[0] * dr**2)
        b_t[0] = 1.0 + 4.0 * l0
        c_t[0] = -4.0 * l0
        d_t[0] = temperature[0]
        for i in range(1, n):
            l_left = k_face[i - 1] * dt / (rho_cp[i] * dr**2)
            l_right = k_face[i] * dt / (rho_cp[i] * dr**2)
            a_t[i] = -(i - 0.5) / i * l_left
            c_t[i] = -(i + 0.5) / i * l_right
            b_t[i] = 1.0 - a_t[i] - c_t[i]
            d_t[i] = temperature[i]
        a_h = 2.0 * n * h * dt / ((n - 0.25) * rho_cp[n] * dr)
        l_n = k_face[n - 1] * dt / (rho_cp[n] * dr**2)
        b_h = 2.0 * (n - 0.5) * l_n / (n - 0.25)
        a_t[n] = -b_h
        b_t[n] = 1.0 + a_h + b_h
        d_t[n] = temperature[n] + a_h * t_bc(current_time)
        updated_t = ns["thomas"](a_t, b_t, c_t, d_t)

        c_face = 0.5 * (prev_c[:-1] + prev_c[1:])
        t_face = 0.5 * (updated_t[:-1] + updated_t[1:])
        lam = d_fun(c_face, t_face) * dt / dr**2
        a_c = np.zeros(n + 1)
        b_c = np.zeros(n + 1)
        c_c = np.zeros(n + 1)
        d_c = np.zeros(n + 1)
        b_c[0] = 1.0 + 4.0 * lam[0]
        c_c[0] = -4.0 * lam[0]
        d_c[0] = concentration[0]
        for i in range(1, n):
            a_c[i] = -(i - 0.5) / i * lam[i - 1]
            c_c[i] = -(i + 0.5) / i * lam[i]
            b_c[i] = 1.0 - a_c[i] - c_c[i]
            d_c[i] = concentration[i]
        a_m = 2.0 * n * hm * dt / ((n - 0.25) * dr)
        b_m = 2.0 * (n - 0.5) * lam[n - 1] / (n - 0.25)
        a_c[n] = -b_m
        b_c[n] = 1.0 + a_m + b_m
        d_c[n] = concentration[n] + a_m * c_bc(current_time)
        updated_c = ns["thomas"](a_c, b_c, c_c, d_c)

        if (
            np.max(np.abs(updated_t - prev_t)) < 1.0e-6
            and np.max(np.abs(updated_c - prev_c)) < 1.0e-8
        ):
            break
    return updated_t, updated_c


def fixed_radius_consistency(ns, end_time_s=14400.0):
    """冻结半径后，比较材料坐标与固定物理坐标结果。"""
    n = ns["N"]
    dt = ns["dt"]
    radius = ns["R0"]
    temp_material = np.full(n + 1, ns["T0"])
    moisture_material = np.full(n + 1, ns["C0"])
    temp_fixed = temp_material.copy()
    moisture_fixed = moisture_material.copy()
    steps = int(end_time_s / dt)
    times = np.empty(steps)
    error_t = np.empty(steps)
    error_c = np.empty(steps)

    for i in range(steps):
        current_time = (i + 1) * dt
        temp_material, moisture_material = thomas_step_material(
            ns, temp_material, moisture_material, current_time, radius
        )
        temp_fixed, moisture_fixed = thomas_step_fixed(
            ns, temp_fixed, moisture_fixed, current_time, radius
        )
        times[i] = current_time
        error_t[i] = np.max(np.abs(temp_material - temp_fixed))
        error_c[i] = np.max(np.abs(moisture_material - moisture_fixed))
    return times, error_t, error_c


def shrinking_mass_balance(ns, end_time_s=250000.0):
    """实际半径函数下，计算材料坐标的质量平衡残差。"""
    n = ns["N"]
    dt = ns["dt"]
    temp = np.full(n + 1, ns["T0"])
    moisture = np.full(n + 1, ns["C0"])
    radius_func = ns["R_of"]
    c_bc = ns["C_bc"]
    hm = ns["hm"]

    dxi = 1.0 / n
    node_index = np.arange(n + 1)
    weights = np.empty(n + 1)
    weights[0] = dxi**2 / 8.0
    weights[1:n] = node_index[1:n] * dxi**2
    weights[n] = (n - 0.25) * dxi**2 / 2.0
    initial_inventory = ns["C0"] / 2.0

    times = []
    residual = []
    outgoing = 0.0
    max_steps = int(end_time_s / dt)
    for i in range(max_steps):
        current_time = (i + 1) * dt
        radius = radius_func(current_time + 0.5 * dt)
        temp, moisture = thomas_step_material(ns, temp, moisture, current_time, radius)
        outgoing += hm / radius * (moisture[-1] - c_bc(current_time)) * dt
        inventory = moisture @ weights
        times.append(current_time)
        residual.append(abs(inventory - initial_inventory + outgoing) / initial_inventory)
        if moisture[0] < ns["C_target"]:
            break
    return np.asarray(times), np.asarray(residual)


def main():
    ns = load_solver_definitions()
    fixed_times, error_t, error_c = fixed_radius_consistency(ns)
    balance_times, residual = shrinking_mass_balance(ns)

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.5,
        }
    )
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.5, 5.1), constrained_layout=True
    )

    # 左图：冻结半径时，变换模型应退化为固定物理坐标模型。
    plot_error_t = np.maximum(error_t, 1.0e-14)
    plot_error_c = np.maximum(error_c, 1.0e-14)
    ax_left.semilogy(fixed_times / 3600.0, plot_error_t, color="#C98C2B", lw=1.8, label="温度场最大差值")
    ax_left.semilogy(fixed_times / 3600.0, plot_error_c, color="#2E6E9E", lw=1.8, label="水分浓度场最大差值")
    ax_left.set(xlim=(0, 4.0), ylim=(1.0e-14, 1.0e-11))
    ax_left.set_xlabel(r"烘干时间 $t/\mathrm{h}$")
    ax_left.set_ylabel("最大绝对差值")
    ax_left.set_title("(a) 固定半径退化一致性")
    ax_left.grid(which="both", axis="y", color="#D9D9D9", linewidth=0.65)
    ax_left.spines[["top", "right"]].set_visible(False)
    ax_left.legend(loc="upper right")
    ax_left.text(
        0.20,
        2.3e-13,
        rf"$\max E_T={sci_math(np.max(error_t))}\,^\circ\mathrm{{C}}$" + "\n" + rf"$\max E_C={sci_math(np.max(error_c))}\,\mathrm{{kg/kg}}$",
        color="#444444",
    )

    # 右图：实际收缩下，材料坐标质量平衡残差。
    display_residual = residual * 1.0e15
    max_residual = float(np.max(residual))
    ax_right.plot(balance_times / 3600.0, display_residual, color="#2E6E9E", lw=1.7)
    ax_right.fill_between(balance_times / 3600.0, display_residual, 0.0, color="#2E6E9E", alpha=0.10)
    ax_right.axhline(max_residual * 1.0e15, color="#C98C2B", linestyle="--", linewidth=1.2)
    ax_right.set(
        xlim=(0, balance_times[-1] / 3600.0),
        ylim=(0, max(3.0, max_residual * 1.0e15 * 1.15)),
    )
    ax_right.set_xlabel(r"烘干时间 $t/\mathrm{h}$")
    ax_right.set_ylabel(r"相对质量平衡残差 $\varepsilon_\xi\,(\times10^{-15})$")
    ax_right.set_title("(b) 收缩条件下的水分质量平衡")
    ax_right.grid(axis="y", color="#D9D9D9", linewidth=0.65)
    ax_right.spines[["top", "right"]].set_visible(False)
    ax_right.text(0.04 * balance_times[-1] / 3600.0, 0.88 * ax_right.get_ylim()[1], rf"最大相对残差 $={sci_math(max_residual)}$", color="#9A661C")

    pdf_path = OUTPUT_DIR / "图13 收缩模型退化一致性与水分质量平衡检验.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"max E_T = {np.max(error_t):.6e} °C")
    print(f"max E_C = {np.max(error_c):.6e} kg/kg")
    print(f"max mass residual = {max_residual:.6e}")
    print(pdf_path)


if __name__ == "__main__":
    main()
