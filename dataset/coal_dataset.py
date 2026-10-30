# -*- coding: utf-8 -*-
"""
数据集
======

原始 Excel → 清洗/特征工程 → 每日图 → 滑动窗口样本 的完整流程：

1. **日期解析**：``_parse_custom_date``（旧数据缺少年份，默认补 2024 年）；
2. **排序 + 填充**：按（地点名称, 埋深, 化验日期）排序后 ``ffill``
   （下午缺失默认等于上午），氧气用「同节点历史均值」填充，其余填 0；
3. **特征工程**：
   - 每个基础特征按（地点名称, 埋深）分组计算一阶差分；
   - 格拉汉姆比值 ``Graham_Ratio = CO / (O2 + 1e-5) × 100``（自然发火经典指标）；
4. **每日图**：见 :mod:`dataset.graph_builder`；
5. **滑动窗口**：过去 ``seq_len`` 天（默认 7）的图序列 → 第 8 天的目标，
   目标为各微观节点 [O2, CO, CO2] 展平向量（索引 [0, 4, 3]，共 18 维）。
"""

from __future__ import annotations

import pickle
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import BASE_FEATURES, TARGET_FEATURE_INDICES
from dataset.graph_builder import DailyGraph, build_single_day_graph

#: 日期解析失败时的兜底策略使用 pandas 解析
_FALLBACK_DATE_PARSER = pd.to_datetime


class CoalMineDataset(Dataset):
    """煤矿束管监测数据集：按天构建两级图，切分为滑动窗口样本。"""

    def __init__(self, excel_path: str, seq_len: int = 7):
        """
        Args:
            excel_path: 清洗后的数据文件路径（.xlsx）
            seq_len: 输入时间步长（默认过去 7 天预测第 8 天）
        """
        self.excel_path = excel_path
        self.seq_len = seq_len
        self.sequences: List[List[DailyGraph]] = []
        self.targets: List[List[float]] = []
        self.daily_graphs: Dict = {}

        self.raw_df = self._load_data()
        self._process_data()       # 核心预处理 + 特征工程
        self._build_daily_graphs() # 构建每日拓扑图
        self._build_sequences()    # 滑动窗口切分

    # ------------------------------------------------------------------
    # 数据加载与预处理
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        print(f"Dataset: 正在读取 {self.excel_path}...")
        return pd.read_excel(self.excel_path)

    @staticmethod
    def _parse_custom_date(date_str) -> object:
        """解析「化验日期」。

        旧数据缺少年份信息，统一假设为 2024 年
        （数据采集自 2024 年，可根据实际数据调整）。
        """
        if isinstance(date_str, str):
            try:
                return datetime.strptime(f"2024-{date_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                return _FALLBACK_DATE_PARSER(date_str)
        return date_str

    def _process_data(self) -> None:
        """数据清洗与特征工程（针对节点特性——埋深——做缺失值填充）。"""

        # 1. 日期解析
        self.raw_df["化验日期"] = self.raw_df["化验日期"].apply(self._parse_custom_date)

        # 2. 先按地点、再按埋深、最后按时间排序，保证同一传感器的数据相邻
        self.raw_df = self.raw_df.sort_values(by=["地点名称", "埋深", "化验日期"])

        # 3. 前向填充：下午缺失默认等于上午的数据
        self.raw_df = self.raw_df.ffill()

        # 4. 氧气缺失：用「同节点自己的历史均值」填充（物理量级更合理）
        if "氧气(%)" in self.raw_df.columns:
            node_mean = self.raw_df.groupby(["地点名称", "埋深"])["氧气(%)"].transform("mean")
            self.raw_df["氧气(%)"] = self.raw_df["氧气(%)"].fillna(node_mean)
            # 兜底：节点全空（几乎不可能）时填 0
            self.raw_df["氧气(%)"] = self.raw_df["氧气(%)"].fillna(0)

        # 5. 其余缺失值填 0
        self.raw_df = self.raw_df.fillna(0)

        # 6. 恢复时间顺序，保证差分计算按时间先后进行
        self.raw_df = self.raw_df.sort_values(by=["化验日期", "地点名称", "埋深"])

        # 7. 一阶差分：按（地点名称, 埋深）分组，避免不同节点间混淆
        for feat in BASE_FEATURES:
            col_name = f"{feat}_diff"
            self.raw_df[col_name] = self.raw_df.groupby(["地点名称", "埋深"])[feat].diff().fillna(0)

        # 8. 格拉汉姆比值（CO/O2 比值，自然发火倾向性经典指标）
        co_val = self.raw_df["一氧化碳(ppm)"]
        o2_val = self.raw_df["氧气(%)"]
        self.raw_df["Graham_Ratio"] = (co_val / (o2_val + 1e-5)) * 100

    # ------------------------------------------------------------------
    # 图构建与序列切分
    # ------------------------------------------------------------------

    def _build_daily_graphs(self) -> None:
        """按日期分组，为每天构建一张两级图。"""
        self.daily_graphs = {
            date: build_single_day_graph(day_data, date)
            for date, day_data in self.raw_df.groupby(self.raw_df["化验日期"].dt.date)
        }

    def _build_sequences(self) -> None:
        """滑动窗口切分：前 ``seq_len`` 天的图序列 → 第 8 天的目标向量。

        个别日期缺少测点记录（例如 2024-03-11 只有 5 个测点），
        这类窗口的节点数与其他窗口不一致，会让目标张量变成不规则数组，
        因此在窗口级别整条跳过，并打印跳过数量。
        """
        print(f"Dataset: 正在构建序列 (Window={self.seq_len})...")
        sorted_dates = sorted(self.daily_graphs.keys())
        self.sequences, self.targets = [], []

        # 以出现次数最多的节点数为准（正常日期应有全部测点）
        node_counts = Counter(len(g.micro_features) for g in self.daily_graphs.values())
        expected_nodes = node_counts.most_common(1)[0][0]

        skipped = 0
        for i in range(len(sorted_dates) - self.seq_len):
            input_dates = sorted_dates[i : i + self.seq_len]
            target_date = sorted_dates[i + self.seq_len]

            input_seq = [self.daily_graphs[d] for d in input_dates]
            target_graph = self.daily_graphs[target_date]

            # 输入或目标节点数异常 → 整条窗口跳过
            if len(target_graph.micro_features) != expected_nodes or any(
                len(g.micro_features) != expected_nodes for g in input_seq
            ):
                skipped += 1
                continue

            # 目标提取（节点顺序与 build_single_day_graph 中的排序一致）
            target_matrix = np.array(target_graph.micro_features)  # [nodes, features]

            # 展平目标: [Node1_O2, Node1_CO, Node1_CO2, Node2_O2, ...]
            target_flat = target_matrix[:, TARGET_FEATURE_INDICES].reshape(-1).tolist()

            self.sequences.append(input_seq)
            self.targets.append(target_flat)

        print(
            f"Dataset: 构建完成，共 {len(self.sequences)} 个样本"
            f"（节点数 != {expected_nodes} 的窗口跳过 {skipped} 个）。"
        )

    # ------------------------------------------------------------------
    # 元信息与持久化
    # ------------------------------------------------------------------

    def get_feature_dimensions(self) -> Dict[str, int]:
        """从第一个样本推断维度信息，供模型构建使用。"""
        if not self.sequences:
            return {}
        sample = self.sequences[0][0]
        return {
            "micro_feature_dim": len(sample.micro_features[0]),
            "macro_feature_dim": len(sample.macro_features[0]),
            "num_micro_nodes": len(sample.micro_features),
            "num_macro_nodes": len(sample.macro_features),
            "target_dim": len(self.targets[0]),
            "selected_feature_dim": len(TARGET_FEATURE_INDICES),
        }

    def save_processed_data(self, path: str) -> None:
        """仅保存核心数据（序列与目标），构建逻辑不持久化。"""
        with open(path, "wb") as f:
            pickle.dump({"sequences": self.sequences, "targets": self.targets}, f)

    @classmethod
    def load_processed_data(cls, path: str) -> "CoalMineDataset":
        """从缓存加载（跳过 Excel 读取与预处理）。

        边索引转换已重构为 :mod:`dataset.graph_builder` 中的模块级函数，
        缓存数据集不再需要任何方法「借用」。
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.sequences = data["sequences"]
        obj.targets = data["targets"]
        return obj

    # ------------------------------------------------------------------
    # PyTorch Dataset 协议
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[DailyGraph], torch.Tensor]:
        return self.sequences[idx], torch.tensor(self.targets[idx], dtype=torch.float32)
