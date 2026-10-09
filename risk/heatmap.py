# -*- coding: utf-8 -*-
"""
风险热力图可视化
================

四类图：

1. **单日热力图**：2×4 布局，右列整体表示回风隅角；
2. **时序总览图**：竖版（Y 轴为日期），最多显示最近 200 天；
3. **30 天详细总览图**：每个格子标注具体风险值；
4. **3D 巷道图**：三条巷道轴线 + 7 个监测点的空间分布与风险着色。

配色统一为 ``RdYlBu_r``（低风险蓝 → 高风险红），分数区间 [0, 15]。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MONITOR_POINTS, RISK_LEVELS

#: 颜色条刻度与等级标签（与 config.RISK_LEVELS 对应）
COLORBAR_TICKS = [t for t, _ in RISK_LEVELS]
COLORBAR_LABELS = [label for _, label in RISK_LEVELS]

#: 文字对比度阈值：分数落在这两个区间外时用白字
_TEXT_COLOR_THRESHOLDS = (3.5, 9.0)


class HeatmapGenerator:
    """把 :class:`risk.scorer.RiskScorer` 的评分结果绘制为各类热力图。"""

    def __init__(self, results: List[Dict], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plot_style()

    @staticmethod
    def _setup_plot_style() -> None:
        """配置绘图风格与中文字体。"""
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["figure.dpi"] = 300

    @staticmethod
    def _pick_text_color(val: float) -> str:
        """按分数选择标注文字颜色，保证与底色有足够对比度。"""
        return "white" if val < _TEXT_COLOR_THRESHOLDS[0] or val > _TEXT_COLOR_THRESHOLDS[1] else "black"

    # ------------------------------------------------------------------
    # 单日热力图
    # ------------------------------------------------------------------

    def create_daily_heatmap(self, date, risk_scores: List[float], heatmap_names: List[str]) -> None:
        """生成单日 2×4 横版热力图。

        布局（最右边一整列显示回风隅角）::

            [皮顺1, 皮顺2, 皮顺3, 回风隅角]
            [轨顺1, 轨顺2, 轨顺3, 回风隅角]
        """
        if len(risk_scores) < 7:
            return

        data = np.array([risk_scores[:3] + [risk_scores[6]], risk_scores[3:6] + [risk_scores[6]]])
        names = np.array([heatmap_names[:3] + [heatmap_names[6]], heatmap_names[3:6] + [heatmap_names[6]]])

        fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
        im = ax.imshow(data, cmap="RdYlBu_r", vmin=0, vmax=15, interpolation="nearest")

        # 颜色条：刻度对应风险等级
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(COLORBAR_TICKS)
        cbar.set_ticklabels(COLORBAR_LABELS, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        ax.set_title(f"煤炭自燃风险热力图 - {date}", fontsize=18, pad=25, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_yticks(range(2))
        ax.set_xticklabels(["1#", "2#", "3#", "回风隅角"], fontsize=14, fontweight="bold")
        ax.set_yticklabels(["皮顺", "轨顺"], fontsize=14, fontweight="bold")
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=14, pad=8)

        # 格子内标注：上半行显示名称，下半行显示分数
        for i in range(2):
            for j in range(4):
                text_color = self._pick_text_color(data[i, j])
                ax.text(j, i - 0.2, names[i, j], ha="center", va="center",
                        color=text_color, fontsize=12, fontweight="bold")
                ax.text(j, i + 0.2, f"{data[i, j]:.2f}", ha="center", va="center",
                        color=text_color, fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"heatmap_{date}.png", bbox_inches="tight", dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # 时序总览图
    # ------------------------------------------------------------------

    def create_time_series_heatmap(self, time_series_data: Dict, max_days: int = 200) -> None:
        """生成竖版时序热力图（Y 轴为日期）。"""
        if not time_series_data:
            return

        dates = sorted(time_series_data.keys())
        if len(dates) > max_days:
            dates = dates[-max_days:]
            print(f"    时序热力图只显示最近{max_days}天的数据")

        matrix = np.array([time_series_data[d] for d in dates])  # (天数, 7)

        # 动态高度：每天约 0.4 英寸，限制在 [10, 100] 英寸
        fig_height = max(10, min(len(dates) * 0.4, 100))
        fig, ax = plt.subplots(figsize=(9, fig_height), dpi=300)

        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=15, aspect="auto", interpolation="nearest")
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.03, aspect=35)
        cbar.set_label("风险系数", fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

        ax.set_title("综采工作面自燃风险演化总览", fontsize=18, pad=20, fontweight="bold")

        ax.set_xticks(range(len(MONITOR_POINTS)))
        ax.set_xticklabels(MONITOR_POINTS, fontsize=13, rotation=0, fontweight="bold")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, pad=10)

        # Y 轴日期标签最多 20 个，避免过密
        step = max(1, len(dates) // 20)
        y_ticks = list(range(0, len(dates), step))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([dates[i].strftime("%m-%d") for i in y_ticks],
                           fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", pad=8)

        # 格子内标注（少于 60 天时显示，只标前 6 个监测点）
        if len(dates) < 60:
            for i in range(len(dates)):
                for j in range(6):
                    val = matrix[i, j]
                    if val > 0.5:
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                color=self._pick_text_color(val), fontsize=10, fontweight="bold")

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.15, right=0.95)
        plt.tight_layout()
        save_path = self.output_dir / "summary_time_series_vertical.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    生成竖版汇总图: {save_path}")

    # ------------------------------------------------------------------
    # 30 天详细总览图
    # ------------------------------------------------------------------

    def create_30days_overview(self, time_series_data: Dict, start_date="2025-03-16") -> None:
        """生成指定日期之后 30 天的详细总览图（每个格子标注具体风险值）。

        Args:
            start_date: 开始日期（"YYYY-MM-DD" 或 datetime），None 时显示最近 30 天
        """
        if not time_series_data:
            return

        dates = sorted(time_series_data.keys())

        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            elif isinstance(start_date, (datetime, pd.Timestamp)):
                start_date = start_date.date()

            start_idx = next((i for i, d in enumerate(dates) if d >= start_date), 0)
            dates = dates[start_idx : start_idx + 30]
            if len(dates) < 30:
                print(f"    从 {start_date} 开始的30天数据不足，只显示 {len(dates)} 天")
            else:
                print(f"    30天总览图显示从 {dates[0]} 到 {dates[-1]} 的数据")
        elif len(dates) > 30:
            dates = dates[-30:]
            print(f"    30天总览图只显示最近30天的数据")

        matrix = np.array([time_series_data[d] for d in dates])  # (天数, 7)

        fig_height = max(12, min(len(dates) * 0.6, 100))
        fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)

        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0, vmax=15, aspect="auto", interpolation="nearest")
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.03, aspect=35)
        cbar.set_label("风险系数", fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12)

        if start_date:
            title = f"综采工作面自燃风险30天演化总览 ({dates[0]} 至 {dates[-1]})"
        else:
            title = "综采工作面自燃风险30天演化总览"
        ax.set_title(title, fontsize=18, pad=20, fontweight="bold")

        ax.set_xticks(range(len(MONITOR_POINTS)))
        ax.set_xticklabels(MONITOR_POINTS, fontsize=13, rotation=0, fontweight="bold")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, pad=10)

        ax.set_yticks(range(len(dates)))
        ax.set_yticklabels([d.strftime("%m-%d") for d in dates], fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", pad=8)

        # 每个格子标注具体风险值（全部 7 个监测点）
        for i in range(len(dates)):
            for j in range(len(MONITOR_POINTS)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=self._pick_text_color(val), fontsize=10, fontweight="bold")

        plt.subplots_adjust(top=0.95, bottom=0.08, left=0.18, right=0.95)
        plt.tight_layout()

        if start_date:
            save_path = self.output_dir / f"30days_overview_{dates[0]}_{dates[-1]}.png"
        else:
            save_path = self.output_dir / "30days_overview_detailed.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    生成30天详细总览图: {save_path}")

    # ------------------------------------------------------------------
    # 3D 巷道图
    # ------------------------------------------------------------------

    def create_3d_visualization(self, risk_scores: List[float], date="") -> None:
        """创建 3D 可视化：三条巷道轴线 + 7 个监测点的空间分布。

        风险评分顺序：``[皮顺1, 皮顺2, 皮顺3, 轨顺1, 轨顺2, 轨顺3, 回风隅角]``。
        """
        try:
            if len(risk_scores) < 7:
                print(f"⚠️ 3D可视化需要7个风险评分，当前提供了{len(risk_scores)}个，跳过绘制")
                return

            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # ---- 巷道主轴线 ----
            # 皮顺（红）：Y=0；轨顺（蓝）：Y=5，与皮顺平行；回风（绿）：皮顺正下方 Z=-3
            ax.plot([0, 10], [0, 0], [0, 0], color="red", linewidth=4, label="皮顺巷道")
            ax.plot([0, 10], [5, 5], [0, 0], color="blue", linewidth=4, label="轨顺巷道")
            ax.plot([0, 10], [0, 0], [-3, -3], color="green", linewidth=4, label="回风巷道")
            # 皮顺 → 回风的垂直通道
            ax.plot([5, 5], [0, 0], [0, -3], color="gray", linewidth=2, linestyle="--")

            # ---- 监测点坐标 ----
            pishun_nodes = [(2, 0, 0), (5, 0, 0), (8, 0, 0)]    # 皮顺1-3（埋深从大到小）
            guishun_nodes = [(2, 5, 0), (5, 5, 0), (8, 5, 0)]   # 轨顺1-3
            huifeng_nodes = [(5, 0, -3)]                        # 回风隅角
            all_nodes = pishun_nodes + guishun_nodes + huifeng_nodes

            cmap, vmin, vmax = "RdBu_r", 0, 15
            for idx, (x, y, z) in enumerate(all_nodes):
                ax.scatter(x, y, z, s=500, c=[risk_scores[idx]], cmap=cmap,
                           vmin=vmin, vmax=vmax, alpha=0.8)
                ax.text(x, y, z + 0.5, f"{MONITOR_POINTS[idx]}\n{risk_scores[idx]:.2f}",
                        ha="center", va="bottom", fontweight="bold")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")

            title = "煤炭自燃发火风险3D可视化"
            if date:
                title += f" - {date}"
            ax.set_title(title, fontsize=16, pad=20)
            ax.legend()

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label("风险系数")

            ax.view_init(elev=30, azim=45)
            plt.tight_layout()

            filename = f"3d_visualization_{date}.png" if date else "3d_visualization.png"
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    3D可视化已保存到: {save_path}")

        except Exception as exc:
            print(f"3D可视化生成失败: {exc}")

    # ------------------------------------------------------------------
    # 完整流程
    # ------------------------------------------------------------------

    def run_all(self, start_date="2025-03-16") -> None:
        """生成全部可视化：逐日热力图 + 3D 图 + 时序总览 + 30 天详图。"""
        time_series_data: Dict = {}

        for res in self.results:
            self.create_daily_heatmap(res["日期"], res["risk_scores"], res["heatmap_names"])
            time_series_data[res["日期"]] = res["risk_scores"]
            self.create_3d_visualization(res["risk_scores"], res["日期"])

        self.create_time_series_heatmap(time_series_data)
        self.create_30days_overview(time_series_data, start_date)
        print("\n[√] 处理完成。")
