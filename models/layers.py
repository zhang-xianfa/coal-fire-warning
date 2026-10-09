# -*- coding: utf-8 -*-
"""
模型子模块
==========

分层图神经网络的各组件：

- :class:`MicroGAT`      微观层图注意力网络（2 层 GAT，多头注意力）
- :class:`MacroGCN`      宏观层图卷积网络（2 层 GCN）
- :class:`EnhancedCrossLayerInteraction`  微观 ↔ 宏观跨层交互
- :class:`TemporalProcessor`  时间序列编码器（2 层 GRU）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class MicroGAT(nn.Module):
    """微观层：在同巷道测点之间做多头图注意力聚合。

    输入特征 = 13 维气体特征 + 8 维节点身份嵌入。
    """

    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        # 第 1 层：多头（拼接输出），第 2 层：单头（平均输出）
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MacroGCN(nn.Module):
    """宏观层：在巷道节点之间做图卷积（2 层 GCN + Dropout）。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.conv2(x, edge_index)
        return x


class EnhancedCrossLayerInteraction(nn.Module):
    """微观 ↔ 宏观跨层交互（残差 + LayerNorm）。

    - **自底向上**：微观特征线性投影后，按隶属关系在所属宏观节点上
      做均值池化，与宏观特征相加；
    - **自顶向下**：融合后的宏观特征投影后广播回所属微观节点，相加。

    说明：旧版构造函数里有一个从未被 ``forward`` 使用的
    ``self.cross_attn``（多头注意力），重构时已删除，前向行为不变。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bottom_up = nn.Linear(dim, dim)
        self.top_down = nn.Linear(dim, dim)
        self.ln_micro = nn.LayerNorm(dim)
        self.ln_macro = nn.LayerNorm(dim)

    def forward(self, micro, macro, cross_link_index):
        micro_indices, macro_indices = cross_link_index[0], cross_link_index[1]
        num_micro = micro.size(0)
        ones = torch.ones(num_micro, 1, device=micro.device)

        # 1. 微观 → 宏观：投影 + 均值池化
        micro_proj = F.relu(self.bottom_up(micro))
        macro_update = torch.zeros_like(macro)
        macro_update.index_add_(0, macro_indices, micro_proj[micro_indices])
        counts = torch.zeros(macro.size(0), 1, device=macro.device)
        counts.index_add_(0, macro_indices, ones[micro_indices])
        macro_combined = self.ln_macro(macro + macro_update / (counts + 1e-6))

        # 2. 宏观 → 微观：投影 + 广播回所属微观节点
        #    每个微观节点只属于一个巷道，counts_micro 恒为 1，
        #    保留统计写法是为了与自底向上保持对称、结构更稳健。
        macro_proj = F.relu(self.top_down(macro_combined))
        micro_update = torch.zeros_like(micro)
        micro_update.index_add_(0, micro_indices, macro_proj[macro_indices])
        counts_micro = torch.zeros_like(ones)
        counts_micro.index_add_(0, micro_indices, ones[micro_indices])
        micro_final = self.ln_micro(micro + micro_update / (counts_micro + 1e-6))

        return micro_final, macro_combined


class TemporalProcessor(nn.Module):
    """时间序列编码器：2 层 GRU，取最后一层隐状态并做 LayerNorm。"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.4)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out, h = self.gru(x)
        return self.ln(h[-1])
