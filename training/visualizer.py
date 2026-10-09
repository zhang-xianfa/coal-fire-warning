# -*- coding: utf-8 -*-
"""
模型可视化
==========

- 模型权重分布直方图；
- 各层重要性（权重 Frobenius 范数）柱状图；
- 两级图结构示意（微观层 / 宏观层 / 跨层连接）。

说明：旧版的 ``visualize_attention_weights`` 依赖模型返回注意力权重，
但当前模型（GATConv 默认不返回注意力，跨层交互中也没有注意力路径）
从未真正产出过结果，重构时已删除。旧版图结构图把宏观层画成
「皮顺/轨顺/回风」3 个节点，与实际模型的 2 个宏观节点不符，已修正。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from dataset.graph_builder import CollatedDay


class ModelVisualizer:
    """模型结构与图结构可视化。"""

    def __init__(self, model, output_dir: str):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    # ------------------------------------------------------------------
    # 权重与层重要性
    # ------------------------------------------------------------------

    def visualize_model_weights(self) -> None:
        """前 9 个可训练权重的分布直方图。"""
        weight_distributions = {
            name: param.data.cpu().numpy().flatten()
            for name, param in self.model.named_parameters()
            if "weight" in name and param.requires_grad
        }
        selected = list(weight_distributions.items())[:9]
        if not selected:
            return

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, (name, weights) in enumerate(selected):
            ax = axes[idx]
            ax.hist(weights, bins=50, alpha=0.7, edgecolor="black")
            ax.set_title(f"{name}\n均值: {np.mean(weights):.4f}", fontsize=10)
            ax.set_xlabel("权重值")
            ax.set_ylabel("频次")
            ax.grid(True, alpha=0.3)
        for idx in range(len(selected), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_weights_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_layer_importance(self) -> None:
        """各层重要性（按权重 Frobenius 范数汇总）。"""
        layer_importance: Dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if "weight" in name and param.requires_grad:
                layer_name = ".".join(name.split(".")[:2])
                layer_importance[layer_name] = layer_importance.get(layer_name, 0) + torch.norm(param.data).item()

        if not layer_importance:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        layers = list(layer_importance.keys())
        importances = list(layer_importance.values())

        sorted_indices = np.argsort(importances)[::-1]
        layers = [layers[i] for i in sorted_indices]
        importances = [importances[i] for i in sorted_indices]

        bars = ax.bar(range(len(layers)), importances, alpha=0.7, color="skyblue")
        ax.set_xlabel("网络层")
        ax.set_ylabel("权重范数（重要性）")
        ax.set_title("各网络层重要性分析")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        for bar, importance in zip(bars, importances):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{importance:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "layer_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 图结构
    # ------------------------------------------------------------------

    def visualize_graph_structure(self, day: CollatedDay, micro_node_ids: List[str]) -> None:
        """可视化两级图结构（微观层 / 宏观层 / 跨层连接）。"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 微观层：按巷道着色（红=皮顺，蓝=轨顺）
        num_micro = day.micro_features.shape[0]
        G_micro = nx.Graph()
        G_micro.add_nodes_from([f"节点_{i}" for i in range(num_micro)])
        if day.micro_edges.shape[1] > 0:
            edges = day.micro_edges.cpu()
            G_micro.add_edges_from(
                (f"节点_{edges[0, i].item()}", f"节点_{edges[1, i].item()}")
                for i in range(edges.shape[1])
            )

        node_colors = [
            "red" if "皮顺" in micro_node_ids[i] else "blue"
            for i in range(num_micro)
        ]
        pos_micro = nx.spring_layout(G_micro, seed=42)
        nx.draw(G_micro, pos_micro, ax=axes[0], node_color=node_colors,
                node_size=100, with_labels=True, font_size=6,
                edge_color="gray", alpha=0.7)
        axes[0].set_title("微观层图结构\n(红:皮顺, 蓝:轨顺)")

        # 2. 宏观层：皮顺 ↔ 轨顺
        G_macro = nx.DiGraph()
        G_macro.add_nodes_from(["皮顺", "轨顺"])
        G_macro.add_edges_from([("皮顺", "轨顺"), ("轨顺", "皮顺")])
        pos_macro = nx.spring_layout(G_macro, seed=42)
        nx.draw(G_macro, pos_macro, ax=axes[1], node_color=["red", "blue"],
                node_size=500, with_labels=True, font_size=12,
                edge_color="gray", alpha=0.7, arrows=True)
        axes[1].set_title("宏观层图结构")

        # 3. 跨层连接
        self._visualize_cross_links(day, axes[2])

        plt.tight_layout()
        plt.savefig(self.output_dir / "graph_structures.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _visualize_cross_links(day: CollatedDay, ax) -> None:
        """绘制微观 → 宏观的跨层连接。"""
        cross_links = day.cross_links.cpu()
        if cross_links.shape[1] == 0:
            return

        num_micro = day.micro_features.shape[0]
        micro_positions = np.random.rand(num_micro, 2)
        macro_positions = np.array([[0.2, 0.5], [0.8, 0.5]])  # 皮顺, 轨顺

        for i in range(cross_links.shape[1]):
            micro_idx, macro_idx = cross_links[0, i].item(), cross_links[1, i].item()
            if micro_idx < len(micro_positions) and macro_idx < len(macro_positions):
                ax.plot([micro_positions[micro_idx, 0], macro_positions[macro_idx, 0]],
                        [micro_positions[micro_idx, 1], macro_positions[macro_idx, 1]],
                        "gray", alpha=0.5, linewidth=1)

        ax.scatter(micro_positions[:, 0], micro_positions[:, 1], c="lightblue",
                   s=100, label="微观节点", alpha=0.7)
        ax.scatter(macro_positions[:, 0], macro_positions[:, 1], c=["red", "blue"],
                   s=200, label="宏观节点", alpha=0.7)

        for pos, label in zip(macro_positions, ["皮顺", "轨顺"]):
            ax.text(pos[0], pos[1] + 0.05, label, ha="center", fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("跨层连接")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def run_all(self, day: CollatedDay, micro_node_ids: List[str]) -> None:
        """一键生成全部模型可视化。"""
        self.visualize_model_weights()
        self.visualize_layer_importance()
        self.visualize_graph_structure(day, micro_node_ids)
