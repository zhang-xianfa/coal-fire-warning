# -*- coding: utf-8 -*-
"""
风险感知损失函数
================

在 MSE 的基础上按「气体 → 风险含义」加权。目标向量按节点展平，
每个节点 3 个指标 ``[O2, CO, CO2]``，因此位置 ``0,3,6,...`` 是 O2、
``1,4,7,...`` 是 CO、``2,5,8,...`` 是 CO2。

- **O2 权重 15.0**：最稳定的环境指标，必须极度精准，是模型可信度的基石；
- **CO 漏报惩罚 2.0**：漏报（真实 > 预测）时权重翻倍，压低漏报风险。
  旧版取 5.0 时模型过于「害怕」漏报，CO 预测值虚高（飙到 10.7 ppm），
  降到 2.0 后模型更理智；
- **CO2 权重 5.0**：权重适中，保持关注。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RiskAwareLoss(nn.Module):
    """加权 MSE：平衡 O2 精度与 CO 漏报风险。"""

    def __init__(self, device, penalty_factor: float = 2.0):
        super().__init__()
        self.device = device
        self.penalty = penalty_factor
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        weights = torch.ones_like(loss)

        # 1. O2（每个节点的第 1 个指标）
        weights[:, 0::3] = 15.0

        # 2. CO：漏报（真实 > 预测）时加重惩罚
        co_weights = torch.where(
            target[:, 1::3] > pred[:, 1::3],
            torch.tensor(self.penalty, device=self.device),
            torch.tensor(1.0, device=self.device),
        )
        weights[:, 1::3] = co_weights

        # 3. CO2（每个节点的第 3 个指标）
        weights[:, 2::3] = 5.0

        return (loss * weights).mean()
