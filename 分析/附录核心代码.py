# -*- coding: utf-8 -*-
"""论文附录用：圆柱径向传热-传质求解的核心算法。

保留有限体积、后向 Euler、Thomas 与 Picard 的计算链；输入输出和绘图
留在各问题脚本中。固定半径与材料坐标只在半径缩放项上不同。
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Boundary:
    """边界函数与传热/传质交换系数。"""
    h: float
    hm: float
    temperature: callable
    concentration: callable


@dataclass
class Properties:
    """把三套附录物性以函数形式注入统一离散框架。"""
    rho: callable
    cp: callable
    k: callable
    diffusivity: callable


def solve_tridiagonal(lower, diagonal, upper, rhs):
    """Thomas 消元；端点不存在的邻接系数置零。"""
    lo = np.asarray(lower, dtype=float).copy()
    diag = np.asarray(diagonal, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    vec = np.asarray(rhs, dtype=float).copy()
    n = diag.size
    scaled_upper = np.zeros(n)
    scaled_rhs = np.zeros(n)
    scaled_upper[0] = hi[0] / diag[0]
    scaled_rhs[0] = vec[0] / diag[0]
    for i in range(1, n):
        pivot = diag[i] - lo[i] * scaled_upper[i - 1]
        scaled_upper[i] = hi[i] / pivot
        scaled_rhs[i] = (vec[i] - lo[i] * scaled_rhs[i - 1]) / pivot
    result = np.empty(n)
    result[-1] = scaled_rhs[-1]
    for i in range(n - 2, -1, -1):
        result[i] = scaled_rhs[i] - scaled_upper[i] * result[i + 1]
    return result


def radial_matrix(face_diffusivity, node_capacity, dt, radius, boundary_rate, boundary_value):
    """按原求解器的半格点有限体积格式组装三对角系统。"""
    node_count = node_capacity.size
    intervals = node_count - 1
    dr = radius / intervals
    lam = face_diffusivity * dt / dr**2
    lower = np.zeros(node_count)
    diagonal = np.zeros(node_count)
    upper = np.zeros(node_count)
    forcing = np.zeros(node_count)

    diagonal[0] = 1.0 + 4.0 * lam[0] / node_capacity[0]
    upper[0] = -4.0 * lam[0] / node_capacity[0]
    for i in range(1, intervals):
        left = (i - 0.5) / i * lam[i - 1] / node_capacity[i]
        right = (i + 0.5) / i * lam[i] / node_capacity[i]
        lower[i] = -left
        upper[i] = -right
        diagonal[i] = 1.0 + left + right

    surface = intervals
    inner = lam[surface - 1] / node_capacity[surface]
    conductance = 2.0 * (surface - 0.5) * inner / (surface - 0.25)
    exchange = 2.0 * surface * boundary_rate * dt / ((surface - 0.25) * dr * node_capacity[surface])
    lower[surface] = -conductance
    diagonal[surface] = 1.0 + conductance + exchange
    forcing[surface] = exchange * boundary_value
    return lower, diagonal, upper, forcing


def fixed_radius_step(temperature, concentration, current_time, radius, dt, props, boundary):
    """固定半径问题的一步 Picard 推进，返回新 T、C 与迭代次数。"""
    new_t = temperature.copy()
    new_c = concentration.copy()
    for iteration in range(1, 51):
        old_t = new_t.copy()
        old_c = new_c.copy()
        rho_cp = props.rho(old_c) * props.cp(old_c)
        k_face = props.k(0.5 * (old_c[:-1] + old_c[1:]))
        thermal_capacity = rho_cp
        a, b, c, forcing = radial_matrix(
            k_face, thermal_capacity, dt, radius, boundary.h, boundary.temperature(current_time)
        )
        new_t = solve_tridiagonal(a, b, c, temperature + forcing)

        c_face = 0.5 * (old_c[:-1] + old_c[1:])
        t_face = 0.5 * (new_t[:-1] + new_t[1:])
        moisture_diffusivity = props.diffusivity(c_face, t_face)
        mass_capacity = np.ones_like(concentration)
        a, b, c, forcing = radial_matrix(
            moisture_diffusivity,
            mass_capacity,
            dt,
            radius,
            boundary.hm,
            boundary.concentration(current_time),
        )
        new_c = solve_tridiagonal(a, b, c, concentration + forcing)
        if np.max(np.abs(new_t - old_t)) < 1e-6 and np.max(np.abs(new_c - old_c)) < 1e-8:
            return new_t, new_c, iteration
    return new_t, new_c, 50


def material_radius_step(temperature, concentration, current_time, radius, dt, props, boundary):
    """材料坐标的一步推进；与固定半径离散同构，只替换 R 为 R(t)。"""
    return fixed_radius_step(temperature, concentration, current_time, radius, dt, props, boundary)


def locate_threshold(previous, current, t_previous, dt, target):
    """用跨越前后的中心浓度线性插值定位达标时刻。"""
    if current >= target:
        return None
    fraction = (target - previous) / (current - previous)
    return t_previous + fraction * dt


def advance_until_dry(temperature, concentration, times, dt, step, target, radius_at, props, boundary):
    """按时间推进并返回首次满足中心浓度判据的时刻。"""
    previous_center = float(concentration[0])
    for time in times:
        temperature, concentration, _ = step(
            temperature, concentration, time, radius_at(time), dt, props, boundary
        )
        crossing = locate_threshold(previous_center, concentration[0], time - dt, dt, target)
        if crossing is not None:
            return crossing, temperature, concentration
        previous_center = float(concentration[0])
    return None, temperature, concentration


# 问题4的非均匀路线在材料质量分数 s 上组装 g_C、g_T：
#   g_C=(4 I^2/R^2) J D/(v^2 ds),  g_T=(4 I^2/R^2) J k/(v ds)。
# 其三对角求解仍调用 solve_tridiagonal，边界交换项仍采用第三类边界。
