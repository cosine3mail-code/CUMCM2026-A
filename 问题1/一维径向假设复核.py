#!/usr/bin/env python3
"""Export Figures 3 and 4 of the Question 1 radial-assumption validation.

This script reads the existing full-precision NPZ result and writes one
two-page PDF containing Figures (a) and (b). It does not rerun the numerical model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plot_question1_radial_validation import axial_average, influence_length, positive_for_log


def parse_arguments(project_root: Path) -> argparse.Namespace:
    default_result = project_root / "计算结果" / "第一题一维径向假设复核"
    default_output = project_root / "plt" / "一维径向假设复核_图三图四"
    parser = argparse.ArgumentParser(
        description="Export standalone PDFs for Figures 3 and 4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_result / "一维二维完整精度对照.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = parse_arguments(project_root)
    if not args.input.is_file():
        raise FileNotFoundError(f"找不到计算结果：{args.input}")

    with np.load(args.input) as data:
        time_s = data["time_s"]
        z_cm = 100.0 * data["axial_position_m"]
        temperature_1d = data["temperature_1d"]
        moisture_1d = data["moisture_1d"]
        temperature_2d = data["temperature_2d"]
        moisture_2d = data["moisture_2d"]
        fraction = float(data["influence_fraction"])
        temperature_span = float(data["temperature_span"])
        moisture_span = float(data["moisture_span"])

    delta_t = np.abs(temperature_2d - temperature_1d[:, :, None])
    delta_c = np.abs(moisture_2d - moisture_1d[:, :, None])
    worst_t_index = int(np.argmax(np.max(delta_t, axis=(1, 2))))
    worst_c_index = int(np.argmax(np.max(delta_c, axis=(1, 2))))

    length_t, axial_t = influence_length(
        temperature_2d[worst_t_index], z_cm, fraction, 1.0e-10
    )
    length_c, axial_c = influence_length(
        moisture_2d[worst_c_index], z_cm, fraction, 1.0e-12
    )
    normalized_t = axial_t / axial_t[0] if axial_t[0] > 0.0 else np.zeros_like(axial_t)
    normalized_c = axial_c / axial_c[0] if axial_c[0] > 0.0 else np.zeros_like(axial_c)

    mid_t = np.max(np.abs(temperature_2d[:, :, -1] - temperature_1d), axis=1)
    mid_c = np.max(np.abs(moisture_2d[:, :, -1] - moisture_1d), axis=1)
    average_t = np.asarray([axial_average(field, z_cm) for field in temperature_2d])
    average_c = np.asarray([axial_average(field, z_cm) for field in moisture_2d])
    average_t_error = np.max(np.abs(average_t - temperature_1d), axis=1)
    average_c_error = np.max(np.abs(average_c - moisture_1d), axis=1)

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 14,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined_pdf = args.output_dir / "问题一一维径向假设复核_图3图4_横向并列.pdf"
    with PdfPages(combined_pdf) as pdf:
        # One landscape page with Figures (a) and (b) side by side.
        fig, (ax3, ax4) = plt.subplots(
            1, 2, figsize=(16.0, 6.4), constrained_layout=True
        )
        ax3.plot(z_cm, normalized_c, linewidth=2.6, label="含水率轴向偏离")
        ax3.plot(z_cm, normalized_t, linewidth=2.6, label="温度轴向偏离")
        ax3.axhline(
            fraction,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"剩余偏离 {100*fraction:g}%",
        )
        ax3.axvline(length_c, color="C0", linestyle=":", linewidth=1.4)
        ax3.axvline(length_t, color="C1", linestyle=":", linewidth=1.4)
        ax3.set_xlim(float(z_cm[0]), float(z_cm[-1]))
        ax3.set_ylim(-0.02, 1.03)
        ax3.set_xlabel("距端面的轴向距离 z / cm")
        ax3.set_ylabel("归一化轴向偏离")
        ax3.set_title("(a) 端面影响沿轴向的衰减")
        ax3.grid(alpha=0.25)
        ax3.legend(loc="upper right", frameon=True, fontsize=13)
        ax3.text(
            0.98,
            0.54,
            f"含水率影响长度：{length_c:.2f} cm\n温度影响长度：{length_t:.2f} cm",
            transform=ax3.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
        )
        # Figure (b): mid-plane and axial-average differences.
        time_min = time_s / 60.0
        mid_c_percent = 100.0 * mid_c / moisture_span
        average_c_percent = 100.0 * average_c_error / moisture_span
        mid_t_percent = 100.0 * mid_t / temperature_span
        average_t_percent = 100.0 * average_t_error / temperature_span
        mid_display_factor = 1.0e4

        ax4.semilogy(
            time_min,
            positive_for_log(mid_c_percent * mid_display_factor),
            linewidth=2.6,
            label=f"含水率中面差异（显示×{mid_display_factor:.0e}）",
        )
        ax4.semilogy(
            time_min,
            positive_for_log(average_c_percent),
            linewidth=2.6,
            linestyle="--",
            label="含水率轴向平均差异",
        )
        ax4.semilogy(
            time_min,
            positive_for_log(mid_t_percent * mid_display_factor),
            linewidth=2.6,
            label=f"温度中面差异（显示×{mid_display_factor:.0e}）",
        )
        ax4.semilogy(
            time_min,
            positive_for_log(average_t_percent),
            linewidth=2.6,
            linestyle="--",
            label="温度轴向平均差异",
        )
        ax4.set_xlabel("时间 / min")
        ax4.set_ylabel("相对于完整变化跨度的差异 / %")
        ax4.set_title("(b) 中面一致性与轴向平均差异随时间变化")
        ax4.set_ylim(1.0e-12, 5.0)
        ax4.grid(alpha=0.25)
        ax4.legend(loc="lower right", frameon=True, fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Combined PDF written to: {combined_pdf.resolve()}")


if __name__ == "__main__":
    main()
