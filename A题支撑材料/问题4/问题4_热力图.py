#!/usr/bin/env python3
"""绘制问题4材料坐标与物理坐标下的水分浓度热力图。

以只读方式载入结果工作簿。材料坐标为 xi=r/R(t)；物理坐标图使用
固定半径坐标，并遮蔽收缩后药材外部的区域。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_XLSX = PROJECT_ROOT / "问题4" / "结果" / "result4.xlsx"
RADIUS_XLSX = PROJECT_ROOT / "附件" / "附件2.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "图"
AUX_PDF = OUTPUT_DIR / "图12  考虑径向收缩的药材水分浓度时空分布.pdf"
AUX_PNG = OUTPUT_DIR / "图12  考虑径向收缩的药材水分浓度时空分布.png"


def read_result(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError("问题四结果文件没有足够数据。")

    # 最后一列单独存放移动表面的数值。
    radius_cm = np.asarray(rows[0][1:-1], dtype=float)
    time_s = np.asarray([row[0] for row in rows[1:]], dtype=float)
    values = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row[1:-1]]
            for row in rows[1:]
        ],
        dtype=float,
    )
    surface = np.asarray(
        [np.nan if row[-1] is None else float(row[-1]) for row in rows[1:]],
        dtype=float,
    )
    if values.shape != (time_s.size, radius_cm.size):
        raise ValueError("问题四结果文件的时间/径向数据维度不一致。")
    if np.any(np.diff(time_s) <= 0.0) or np.any(np.diff(radius_cm) <= 0.0):
        raise ValueError("时间和径向坐标必须严格递增。")
    return time_s, radius_cm, values, surface


def read_radius_law(path: Path) -> tuple[np.ndarray, np.ndarray]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(min_row=2, values_only=True))
    finally:
        workbook.close()
    data = np.asarray(
        [(float(row[0]), float(row[1])) for row in rows if row[0] is not None],
        dtype=float,
    )
    if data.size == 0 or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError("附件2没有有效且递增的半径数据。")
    return data[:, 0], data[:, 1]


def cell_edges(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    interior = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([lower], interior, [upper]))


def add_threshold_contour(ax, x_grid, y_grid, field, *, threshold=0.15):
    levels = np.array([0.30, 0.60, 0.90, 1.20, 1.50, 1.80, 2.10, 2.40])
    finite = np.asarray(field)[np.isfinite(field)]
    levels = levels[(levels > finite.min()) & (levels < finite.max())]
    if levels.size:
        regular = ax.contour(
            x_grid, y_grid, field, levels=levels,
            colors="white", linewidths=0.8, alpha=0.86,
        )
        labels = ax.clabel(regular, inline=True, fontsize=8, fmt="%.2f", colors="white")
        for label in labels:
            label.set_fontweight("bold")
    if finite.min() <= threshold <= finite.max():
        marked = ax.contour(
            x_grid, y_grid, field, levels=[threshold],
            colors="#d62728", linewidths=2.2, alpha=1.0,
        )
        labels = ax.clabel(
            marked, inline=True, fontsize=9,
            fmt={threshold: "0.15"}, colors="#d62728",
        )
        for label in labels:
            label.set_fontweight("bold")
            label.set_bbox(dict(facecolor="white", edgecolor="#d62728", alpha=0.85, pad=1.5))


def style_colorbar(fig, image, ax):
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", pad=0.025)
    colorbar.set_label("干基含水率 / (kg/kg)", labelpad=8)
    colorbar.set_ticks([0.15, 0.50, 1.00, 1.50, 2.00, 2.50])
    colorbar.ax.tick_params(length=3)
    for tick, value in zip(colorbar.ax.get_yticklabels(), colorbar.get_ticks()):
        if np.isclose(value, 0.15):
            tick.set_color("#d62728")
            tick.set_fontweight("bold")


def make_ticks(end_time_h: float) -> np.ndarray:
    interval = 6.0 if end_time_h <= 36.0 else 12.0
    ticks = np.arange(0.0, end_time_h + 1.0e-12, interval)
    if ticks.size == 0 or not np.isclose(ticks[-1], end_time_h):
        ticks = np.append(ticks, end_time_h)
    return ticks


def prepare_data():
    time_s, radius_cm, values, surface = read_result(RESULT_XLSX)
    radius_time_s, radius_law_cm = read_radius_law(RADIUS_XLSX)
    current_radius_cm = np.interp(time_s, radius_time_s, radius_law_cm)

    # 补入 t=0 的均匀初始状态，使左边界完整。
    if time_s[0] > 0.0:
        time_s = np.insert(time_s, 0, 0.0)
        values = np.vstack((np.full((1, values.shape[1]), 2.55), values))
        surface = np.insert(surface, 0, 2.55)
        # 补入点使用实测初始半径，不使用首个记录时刻（t=60 s）的半径。
        current_radius_cm = np.insert(
            current_radius_cm, 0, float(radius_law_cm[0])
        )

    time_h = time_s / 3600.0
    # 将固定半径采样转换为 xi，并逐时刻插值。
    xi_display = np.linspace(0.0, 1.0, 241)
    normalized = np.empty((time_s.size, xi_display.size), dtype=float)
    for i, (row, radius_cm_now, surface_value) in enumerate(
        zip(values, current_radius_cm, surface)
    ):
        valid = np.isfinite(row) & (radius_cm <= radius_cm_now + 1.0e-10)
        xi = np.concatenate((radius_cm[valid] / radius_cm_now, [1.0]))
        field = np.concatenate((row[valid], [surface_value]))
        order = np.argsort(xi)
        xi, field = xi[order], field[order]
        unique_xi, unique_indices = np.unique(xi, return_index=True)
        normalized[i] = np.interp(xi_display, unique_xi, field[unique_indices])
    # 平滑固定半径下的显示结果，同时遮蔽药材外部；移动边界处采用表面值。
    physical_display_radius = np.linspace(
        float(radius_cm[0]), float(radius_cm[-1]), 401
    )
    physical_display = np.full(
        (time_s.size, physical_display_radius.size), np.nan, dtype=float
    )
    for i, (row, radius_cm_now, surface_value) in enumerate(
        zip(values, current_radius_cm, surface)
    ):
        valid = np.isfinite(row) & (radius_cm <= radius_cm_now + 1.0e-10)
        x = np.concatenate((radius_cm[valid], [radius_cm_now]))
        y = np.concatenate((row[valid], [surface_value]))
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique_x, unique_indices = np.unique(x, return_index=True)
        inside = physical_display_radius <= radius_cm_now + 1.0e-10
        physical_display[i, inside] = np.interp(
            physical_display_radius[inside], unique_x, y[unique_indices]
        )
    return (
        time_h,
        physical_display_radius,
        physical_display,
        current_radius_cm,
        normalized,
        xi_display,
    )


def main() -> None:
    time_h, radius_cm, physical_values, current_radius_cm, normalized, xi_display = prepare_data()
    end_time_h = float(time_h[-1])
    time_edges = cell_edges(time_h, 0.0, end_time_h)
    xi_edges = cell_edges(xi_display, 0.0, 1.0)
    physical_edges = cell_edges(radius_cm, float(radius_cm.min()), float(radius_cm.max()))
    time_grid, xi_grid = np.meshgrid(time_edges, xi_edges)
    physical_time_grid, physical_grid = np.meshgrid(time_edges, physical_edges)
    time_nodes, xi_nodes = np.meshgrid(time_h, xi_display)
    time_nodes_physical, radius_nodes = np.meshgrid(time_h, radius_cm)

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    })

    # 固定物理坐标下，移动边界外部以灰色显示。
    physical_field = np.ma.masked_invalid(physical_values.T)
    cmap = plt.get_cmap("cividis_r").copy()
    cmap.set_bad("#E6E6E6")
    fig, ax = plt.subplots(figsize=(12.2, 5.8), constrained_layout=True)
    image = ax.pcolormesh(
        physical_time_grid, physical_grid, physical_field, shading="flat",
        cmap=cmap, norm=colors.Normalize(vmin=0.15, vmax=2.55), rasterized=True,
    )
    contour_field = np.ma.masked_invalid(physical_values.T)
    add_threshold_contour(ax, time_nodes_physical, radius_nodes, contour_field)
    ax.plot(time_h, current_radius_cm, color="black", linewidth=1.8, label="药材表面 R(t)")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("到药材中心的距离 / cm")
    ax.set_xlim(0.0, end_time_h)
    ax.set_ylim(float(radius_cm.min()), float(radius_cm.max()))
    ax.set_xticks(make_ticks(end_time_h))
    ax.set_yticks(np.arange(0.0, 2.01, 0.5))
    ax.tick_params(direction="in", length=4)
    ax.grid(False)
    ax.legend(loc="upper right", frameon=True)
    style_colorbar(fig, image, ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUX_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(AUX_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"辅助图 PDF 已保存至：{AUX_PDF}")


if __name__ == "__main__":
    main()
