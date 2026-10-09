# -*- coding: utf-8 -*-
"""
分层图神经网络主模型
====================

对每个时间步执行「空间建模 + 跨层融合」，整个序列交给 GRU 做时序建模：

1. **微观层**：MicroGAT 聚合 6 个测点的特征（13 维气体特征 + 8 维节点身份嵌入）；
2. **宏观层**：MacroGCN 聚合 2 条巷道的特征；
3. **跨层交互**：微观 ↔ 宏观双向消息传递（均值池化上送 / 广播下传）；
4. **时序编码**：拼接 [全部微观特征, 全部宏观特征] 送入 2 层 GRU；
5. **预测头**：MLP 输出 ``num_micro × 3`` 维目标（各测点的 O2/CO/CO2）。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.layers import (
    EnhancedCrossLayerInteraction,
    MacroGCN,
    MicroGAT,
    TemporalProcessor,
)


class HierarchicalGNN(nn.Module):
    """「微观 GAT + 宏观 GCN + 跨层交互 + GRU 时序」分层时空预测模型。"""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        micro_dim = config["micro_feature_dim"]
        macro_dim = config["macro_feature_dim"]
        target_dim = config["target_dim"]
        hidden_dim = config.get("hidden_dim", 64)

        self.num_micro = config["num_micro_nodes"]
        self.num_macro = config["num_macro_nodes"]

        # 节点身份嵌入（每个测点一个可学习的 8 维向量）
        self.node_embedding = nn.Embedding(self.num_micro, 8)

        self.micro_gnn = MicroGAT(micro_dim + 8, hidden_dim)
        self.macro_gnn = MacroGCN(macro_dim, hidden_dim)
        self.cross_layer = EnhancedCrossLayerInteraction(hidden_dim)

        # 每个时间步的拼接特征 = 全部微观隐状态 + 全部宏观隐状态
        temporal_input_dim = self.num_micro * hidden_dim + self.num_macro * hidden_dim
        self.temporal = TemporalProcessor(temporal_input_dim, hidden_dim * 2)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, batch_sequences):
        """batch_sequences: 形状 [batch, seq_len] 的 CollatedDay 列表，
        由 collate_fn 归一化并转移到设备上。"""
        batch_size = len(batch_sequences)
        seq_len = len(batch_sequences[0])
        device = next(self.parameters()).device

        # 节点 ID 索引 [0, 1, ..., num_micro-1] → 嵌入 [num_micro, 8]
        node_ids = torch.arange(self.num_micro, device=device)
        node_embs = self.node_embedding(node_ids)

        rnn_inputs = []

        for b in range(batch_size):
            seq_feats = []
            for t in range(seq_len):
                day = batch_sequences[b][t]
                micro_in = torch.cat([day.micro_features, node_embs], dim=1)

                # 1. 空间建模：微观 GAT + 宏观 GCN
                micro_h = self.micro_gnn(micro_in, day.micro_edges)
                macro_h = self.macro_gnn(day.macro_features, day.macro_edges)

                # 2. 跨层交互：双向消息传递
                micro_h, macro_h = self.cross_layer(micro_h, macro_h, day.cross_links)

                # 3. 展平拼接，作为该时间步的时空特征
                combined = torch.cat([micro_h.reshape(-1), macro_h.reshape(-1)])
                seq_feats.append(combined)

            rnn_inputs.append(torch.stack(seq_feats))

        # 4. 时序编码 + 预测
        rnn_tensor = torch.stack(rnn_inputs)
        gru_out = self.temporal(rnn_tensor)
        preds = self.predictor(gru_out)
        return preds


class ModelManager:
    """模型的构建、保存与加载。"""

    @staticmethod
    def create_model_from_dataset(dataset, hidden_dim: int = 64):
        """根据数据集的维度信息构建模型。

        Returns:
            (model, config)
        """
        dims = dataset.get_feature_dimensions()
        config = {
            "micro_feature_dim": dims["micro_feature_dim"],
            "macro_feature_dim": dims["macro_feature_dim"],
            "target_dim": dims["target_dim"],
            "num_micro_nodes": dims["num_micro_nodes"],
            "num_macro_nodes": dims["num_macro_nodes"],
            "hidden_dim": hidden_dim,
        }
        return HierarchicalGNN(config), config

    @staticmethod
    def save_model(model, path: str, config: Dict) -> None:
        torch.save({"state_dict": model.state_dict(), "config": config}, path)

    @staticmethod
    def load_model(path: str, device):
        ckpt = torch.load(path, map_location=device)
        model = HierarchicalGNN(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
