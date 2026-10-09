# -*- coding: utf-8 -*-
"""
数据标准化
==========

三路互不共享参数的 StandardScaler：

- 微观节点特征（13 维）
- 宏观节点特征（13 维）
- 目标向量（18 维）

只在**训练集**上拟合参数，杜绝验证/测试信息泄漏。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dataset.graph_builder import DailyGraph


class Normalizer:
    """微观 / 宏观 / 目标三路标准化。"""

    def __init__(self):
        self.micro_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit(self, dataset, indices) -> None:
        """只用 ``indices``（训练集下标）中的样本拟合。"""
        print("Trainer: 正在计算归一化参数...")
        micros, macros, tgts = [], [], []
        for idx in indices:
            seq, tgt = dataset[idx]
            for day in seq:
                micros.extend(day.micro_features)
                macros.extend(day.macro_features)
            tgts.append(tgt.numpy())

        self.micro_scaler.fit(np.array(micros))
        self.macro_scaler.fit(np.array(macros))
        self.target_scaler.fit(np.array(tgts))

    def transform_day(self, day: DailyGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """单日特征标准化，返回 CPU 张量 (micro, macro)。"""
        micro = torch.FloatTensor(self.micro_scaler.transform(np.array(day.micro_features)))
        macro = torch.FloatTensor(self.macro_scaler.transform(np.array(day.macro_features)))
        return micro, macro

    def transform_target(self, tgt: torch.Tensor) -> torch.Tensor:
        """标准化单个目标向量（[18] → [18]）。"""
        return torch.FloatTensor(self.target_scaler.transform(tgt.reshape(1, -1)).flatten())

    def inverse_target(self, pred: torch.Tensor) -> np.ndarray:
        """把模型输出还原到真实量纲。"""
        return self.target_scaler.inverse_transform(pred.cpu().numpy())
