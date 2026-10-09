# -*- coding: utf-8 -*-
"""
风险评分
========

「熵权法基础得分 × 7 日趋势因子 × CO 浓度压制」三步风险模型：

1. **特征工程**：逐监测点计算 6 个特征 —— CO 浓度、CO2 浓度、
   O2 消耗量、CO 日增量、CO2 日增量、格拉汉姆比值；
2. **熵权法**：按特征的信息量自动赋权，加权求和得到基础得分并归一化到 [0, 1]；
3. **趋势修正**：基于最近 7 天气体变化率与动态权重（浓度越高/越低
   权重越大），经 tanh 映射得到 [0.8, 1.5] 的趋势因子；
4. **得分映射**：基础得分 × 趋势因子 × 15，再按 CO 浓度压制
   （CO < 24ppm 时指数强压制，≥ 24ppm 不压制），最终限幅在 [0, 15]。
"""

from __future__ import annotations

import math
import os
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from config import MONITOR_POINTS
from risk.entropy_weight import entropy_weights

warnings.filterwarnings("ignore")

#: 风险模型使用的气体指标列
GAS_COLUMNS = ["氧气(%)", "二氧化碳(%)", "一氧化碳(ppm)"]

#: 6 个风险特征（顺序即特征向量顺序）
FEATURE_NAMES = ["CO", "CO2", "O2消耗", "CO日增量", "CO2日增量", "格拉汉姆比值"]

#: 大气默认值（监测点完全无数据时使用）
ATMOSPHERE_DEFAULTS = {"氧气(%)": 20.9, "二氧化碳(%)": 0.04, "一氧化碳(ppm)": 0.0}


class RiskScorer:
    """煤炭自燃风险评分器：数据整理 + 熵权法评分 + CO 超限统计。"""

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.daily_data: pd.DataFrame | None = None  # 整理后的逐日逐监测点数据
        self.dates: List = []                        # 日期列表

    # ------------------------------------------------------------------
    # 日期解析与数据整理
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str) -> object:
        """解析「化验日期」。

        数据跨年存放且原始字符串缺少年份，按月份补全：
        **1-5 月为 2025 年，6-12 月为 2024 年**（与数据的实际采集时间对应）。
        """
        if isinstance(date_str, str):
            try:
                parts = date_str.split("-")
                if len(parts) >= 1:
                    month = int(parts[0])
                    year = 2025 if 1 <= month <= 5 else 2024
                    full_date_str = f"{year}-{date_str}"
                    try:
                        return datetime.strptime(full_date_str, "%Y-%m-%d %H:%M")
                    except ValueError:
                        return datetime.strptime(full_date_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return pd.NaT
        return date_str

    def prepare_daily_data(self) -> None:
        """加载数据 → 清洗 → 插值 → 整理为 7 个监测点的逐日序列。"""
        print(f"[-] 正在加载数据: {self.excel_path}")

        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"找不到文件: {self.excel_path}")

        # 1. 读取 + 日期解析
        self.raw_df = pd.read_excel(self.excel_path)
        self.raw_df["化验日期"] = self.raw_df["化验日期"].apply(self._parse_date)
        self.raw_df.dropna(subset=["化验日期"], inplace=True)
        self.raw_df["日期"] = self.raw_df["化验日期"].dt.date

        # 2. 数值化
        for col in GAS_COLUMNS + ["埋深"]:
            self.raw_df[col] = pd.to_numeric(self.raw_df[col], errors="coerce")

        # 3. 按地点分组线性插值补全缺失
        self.raw_df[GAS_COLUMNS] = self.raw_df.groupby("地点名称")[GAS_COLUMNS].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        self.raw_df.fillna(0, inplace=True)

        # 4. 按日均值聚合
        grouped = (
            self.raw_df.groupby(["日期", "地点名称", "埋深"])[GAS_COLUMNS]
            .mean()
            .reset_index()
        )
        unique_dates = sorted(grouped["日期"].unique())

        # 5. 结构化整理：为每个监测点建立完整的逐日序列
        monitor_data: Dict[str, List[Dict]] = {point: [] for point in MONITOR_POINTS}

        for date in unique_dates:
            daily_group = grouped[grouped["日期"] == date]

            # 皮顺 / 轨顺：按埋深从大到小对应 1#、2#、3#
            for loc_name in ["皮顺", "轨顺"]:
                location_data = daily_group[daily_group["地点名称"] == loc_name].sort_values(
                    "埋深", ascending=False
                )
                for i in range(3):
                    if i < len(location_data):
                        row = location_data.iloc[i]
                        monitor_data[f"{loc_name}{i+1}"].append({
                            "日期": date,
                            "埋深": row["埋深"],
                            "氧气(%)": row["氧气(%)"],
                            "二氧化碳(%)": row["二氧化碳(%)"],
                            "一氧化碳(ppm)": row["一氧化碳(ppm)"],
                        })

            # 回风隅角：取当天第一条「回风」数据
            hf_data = daily_group[daily_group["地点名称"].str.contains("回风", na=False)]
            if not hf_data.empty:
                row = hf_data.iloc[0]
                monitor_data["回风隅角"].append({
                    "日期": date,
                    "埋深": row["埋深"],
                    "氧气(%)": row["氧气(%)"],
                    "二氧化碳(%)": row["二氧化碳(%)"],
                    "一氧化碳(ppm)": row["一氧化碳(ppm)"],
                })

        # 6. 逐监测点补全缺失日期（插值）
        processed_data: List[Dict] = []
        for point in MONITOR_POINTS:
            point_data = monitor_data[point]
            if not point_data:
                # 该监测点完全没有数据：整段用大气默认值
                for date in unique_dates:
                    processed_data.append({
                        "日期": date,
                        "地点名称": point[:2],
                        "埋深": 0,
                        "监测点": point,
                        "热力图显示名称": f"{point[:2]}{point[2:]}-无数据",
                        **ATMOSPHERE_DEFAULTS,
                    })
                continue

            date_df = pd.DataFrame(point_data).set_index("日期")
            # 重索引到完整日期序列并线性插值
            full_date_df = date_df.reindex(unique_dates).interpolate(
                method="linear", limit_direction="both"
            )
            # 埋深恒定：用第一个有效值回填
            fill_depth = full_date_df["埋深"].dropna().iloc[0] if not full_date_df["埋深"].isnull().all() else 0
            full_date_df["埋深"].fillna(fill_depth, inplace=True)

            for date in unique_dates:
                row = full_date_df.loc[date]
                processed_data.append({
                    "日期": date,
                    "地点名称": point[:2],
                    "埋深": row["埋深"],
                    "监测点": point,
                    "热力图显示名称": f"{point[:2]}-{row['埋深']:.1f}m",
                    "氧气(%)": row["氧气(%)"],
                    "二氧化碳(%)": row["二氧化碳(%)"],
                    "一氧化碳(ppm)": row["一氧化碳(ppm)"],
                })

        self.daily_data = pd.DataFrame(processed_data).sort_values(["日期", "监测点"])
        self.dates = sorted(self.daily_data["日期"].unique())
        print(f"[+] 数据加载完成，共 {len(self.dates)} 天的数据")

    # ------------------------------------------------------------------
    # 风险评分
    # ------------------------------------------------------------------

    def _calculate_trend_factor(self, current_date, location: str, history_buffer: Dict) -> float:
        """7 日综合趋势因子（基于动态权重的变化率）。

        - CO / CO2 浓度越高权重越大；O2 浓度越低权重越大；
        - 趋势得分 = CO 增量 + CO2 增量 − O2 增量（O2 下降意味着风险）；
        - tanh 非线性映射后缩放，限制在 [0.8, 1.5]。
        """
        if location not in history_buffer:
            return 1.0

        history = sorted(history_buffer[location], key=lambda x: x["date"])
        recent = history[-7:]  # 最近 7 天
        n = len(recent)
        if n < 7:  # 数据点太少，无法判断趋势
            return 1.0

        co_vals = np.array([r["co"] for r in recent])
        co2_vals = np.array([r["co2"] for r in recent])
        o2_vals = np.array([r["o2"] for r in recent])

        # 平均变化率（用均值比单日差分更稳定）
        co_change = np.mean(np.diff(co_vals[-7:]))
        co2_change = np.mean(np.diff(co2_vals[-7:]))
        o2_change = np.mean(np.diff(o2_vals[-7:]))

        # 动态权重：浓度越偏离常态，权重越大
        co_weight = 0.3 + 0.7 * (co_vals[-1] / (co_vals[-1] + 24))       # 0.3-1.0
        co2_weight = 0.3 + 0.7 * (co2_vals[-1] / (co2_vals[-1] + 5))     # 0.3-1.0
        o2_weight = 0.3 + 0.7 * ((20.9 - o2_vals[-1]) / 20.9)            # 0.3-1.0
        total_weight = co_weight + co2_weight + o2_weight
        co_weight, co2_weight, o2_weight = co_weight / total_weight, co2_weight / total_weight, o2_weight / total_weight

        # 综合趋势：CO 上升 / CO2 上升 / O2 下降都增加风险
        trend_score = co_change * co_weight + co2_change * co2_weight - o2_change * o2_weight

        # tanh 限幅后缩放：0.3 为最大调整幅度
        normalized_trend = np.tanh(trend_score * 0.2)
        trend_factor = 1.0 + normalized_trend * 0.3
        return max(0.8, min(trend_factor, 1.5))

    @staticmethod
    def _map_score(base_score: float, trend_factor: float, co_value: float, co2_value: float, o2_value: float) -> float:
        """风险得分映射与 CO 指数平滑压制。

        基础得分 → 趋势修正 → 缩放至 [0, 15] → CO 压制 → 限幅。
        压制规则：CO < 24ppm 指数强压制（低浓度下分数显著拉低），
        CO ≥ 24ppm 不压制。
        """
        trended_score = base_score * trend_factor
        scaled_score = trended_score * 15.0

        if co_value < 24:
            x = co_value / 24
            alpha_base = 0.1
            curvature = 1.2
            suppress_coeff = alpha_base + (1 - alpha_base) * (1 - math.exp(-curvature * x))
        else:
            suppress_coeff = 1.0

        return max(0.0, min(scaled_score * suppress_coeff, 15.0))

    def calculate_risk_scores(self) -> List[Dict]:
        """计算逐日风险评分。

        Returns:
            结果列表，每项 ``{日期, risk_scores(7 点得分), heatmap_names}``
        """
        print("\n[-] 开始计算风险模型 (熵权法 + 7日线性回归趋势)...")

        # 1. 特征工程
        features: List[List[float]] = []
        records: List[Dict] = []
        previous_day_map: Dict[str, Dict] = {}  # 监测点 → 前一天气体浓度

        for date in self.dates:
            daily_df = self.daily_data[self.daily_data["日期"] == date]
            today_data_map = {}  # 当天数据暂存，避免同一天内监测点间串用

            for _, row in daily_df.iterrows():
                location = row["监测点"]
                co, co2, o2 = row["一氧化碳(ppm)"], row["二氧化碳(%)"], row["氧气(%)"]

                # 前一天数据（首次出现时用当天数据兜底）
                prev = previous_day_map.get(location, {"co": co, "co2": co2, "o2": o2})

                # 4 个关键特征
                co_delta = max(0, co - prev["co"])        # CO 日增量（只取正值）
                co2_delta = max(0, co2 - prev["co2"])     # CO2 日增量（只取正值）
                o2_consumed = max(0, prev["o2"] - o2)     # O2 消耗量（只取正值）

                # 格拉汉姆比值：CO / O2 消耗（自燃倾向性指标）
                graham = co / (o2_consumed * 10000) if o2_consumed > 0.01 else 0

                today_data_map[location] = {"co": co, "co2": co2, "o2": o2}

                features.append([co, co2, o2_consumed, co_delta, co2_delta, graham])
                records.append({
                    "date": date,
                    "loc": location,
                    "display_name": row["热力图显示名称"],
                    "co": co, "co2": co2, "o2": o2,
                })

            # 当天全部处理完再统一更新，保证每个点用的都是真正的前一天数据
            previous_day_map.update(today_data_map)

        # 2. 熵权法 + 基础得分归一化到 [0, 1]
        feature_matrix = np.array(features)
        weights = entropy_weights(feature_matrix)
        print(
            f"    权重计算: CO={weights[0]:.2f}, CO2={weights[1]:.2f}, "
            f"O2耗={weights[2]:.2f}, CO日增={weights[3]:.2f}, "
            f"CO2日增={weights[4]:.2f}, 格拉汉姆={weights[5]:.2f}"
        )

        min_features = feature_matrix.min(axis=0)
        max_features = feature_matrix.max(axis=0)
        normalized_matrix = (feature_matrix - min_features) / (max_features - min_features + 1e-8)

        base_scores = np.dot(normalized_matrix, weights)
        base_min, base_max = base_scores.min(), base_scores.max()
        normalized_base_scores = (base_scores - base_min) / (base_max - base_min + 1e-8)

        # 3. 趋势修正与得分映射
        scored_df = pd.DataFrame(records)
        scored_df["base_score"] = normalized_base_scores

        # 显示顺序：皮顺1-3、轨顺1-3 优先，其余排后
        order_map = {
            f"{t}{i}": (t_idx, i)
            for t_idx, t in enumerate(["皮顺", "轨顺"])
            for i in range(1, 4)
        }

        final_results: List[Dict] = []
        all_final_scores: List[float] = []
        history_buffer: Dict[str, List[Dict]] = {}

        for date in self.dates:
            daily_slice = scored_df[scored_df["date"] == date]
            sorted_rows = sorted(
                daily_slice.to_dict("records"), key=lambda x: order_map.get(x["loc"][:3], (9, 9))
            )

            daily_risk_scores, daily_names = [], []
            for row in sorted_rows:
                location = row["loc"]

                history_buffer.setdefault(location, []).append({
                    "date": row["date"], "co": row["co"], "co2": row["co2"], "o2": row["o2"],
                })

                trend_factor = self._calculate_trend_factor(date, location, history_buffer)
                final_risk = self._map_score(
                    row["base_score"], trend_factor, row["co"], row["co2"], row["o2"]
                )

                daily_risk_scores.append(final_risk)
                all_final_scores.append(final_risk)
                daily_names.append(row["display_name"])

            final_results.append({
                "日期": date,
                "risk_scores": daily_risk_scores,
                "heatmap_names": daily_names,
            })

        if all_final_scores:
            print(
                f"    风险得分统计: 最高分={max(all_final_scores):.2f}, "
                f"平均分={np.mean(all_final_scores):.2f}, 得分范围=[0, 15]"
            )

        return final_results

    # ------------------------------------------------------------------
    # CO 超限统计
    # ------------------------------------------------------------------

    def stat_co_exceed(self) -> Dict:
        """统计 CO 浓度超过 24ppm（预警阈值）的情况。"""
        if self.daily_data is None:
            self.prepare_daily_data()

        total_measurements = len(self.daily_data)
        exceed_data = self.daily_data[self.daily_data["一氧化碳(ppm)"] > 24]
        total_exceed = len(exceed_data)
        exceed_rate = (total_exceed / total_measurements * 100) if total_measurements > 0 else 0

        date_stats = exceed_data.groupby("日期").size().to_dict()
        monitor_stats = exceed_data.groupby("监测点").size().to_dict()

        max_co = exceed_data["一氧化碳(ppm)"].max() if not exceed_data.empty else 0
        min_co = exceed_data["一氧化碳(ppm)"].min() if not exceed_data.empty else 0
        avg_co = exceed_data["一氧化碳(ppm)"].mean() if not exceed_data.empty else 0

        # 打印统计结果
        print("\n" + "=" * 60)
        print("一氧化碳浓度超过24ppm统计结果")
        print("=" * 60)
        print(f"总测量次数: {total_measurements:8d}")
        print(f"总超限次数: {total_exceed:8d}")
        print(f"超限率: {exceed_rate:12.2f}%")
        if not exceed_data.empty:
            print(f"\n超限浓度统计:")
            print(f"最高浓度: {max_co:10.2f} ppm")
            print(f"最低浓度: {min_co:10.2f} ppm")
            print(f"平均浓度: {avg_co:10.2f} ppm")

        if date_stats:
            print(f"\n按日期统计:")
            for date, count in sorted(date_stats.items(), key=lambda x: x[0]):
                print(f"  {date}: {count}次")

        if monitor_stats:
            print(f"\n按监测点统计:")
            for monitor, count in sorted(monitor_stats.items(), key=lambda x: x[1], reverse=True):
                monitor_total = len(self.daily_data[self.daily_data["监测点"] == monitor])
                monitor_rate = (count / monitor_total * 100) if monitor_total > 0 else 0
                print(f"  {monitor}: {count}次 (超限率: {monitor_rate:.1f}%)")

        if not exceed_data.empty:
            print(f"\n详细记录: 共{len(exceed_data)}条")
            print("\n最近5条超限记录:")
            for _, row in exceed_data.tail(5).iterrows():
                print(f"  {row['日期']} {row['监测点']}: {row['一氧化碳(ppm)']:.2f} ppm")
        print("=" * 60)

        return {
            "total_exceed": total_exceed,
            "total_measurements": total_measurements,
            "exceed_rate": exceed_rate,
            "max_co": max_co,
            "min_co": min_co,
            "avg_co": avg_co,
            "date_stats": date_stats,
            "monitor_stats": monitor_stats,
        }
