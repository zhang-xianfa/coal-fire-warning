# -*- coding: utf-8 -*-
"""
每日图结构构建
==============

把某一天的监测记录构建为「微观—宏观」两级图：

- **微观层**：6 个测点（皮顺/轨顺 × 3 个埋深），节点特征 13 维
  （6 基础特征 + 6 一阶差分 + 1 格拉汉姆比值）；
- **宏观层**：2 条巷道（皮顺、轨顺），节点特征为所属微观节点的均值；
- **跨层连边**：每个微观节点指向其所属巷道的宏观节点。

边索引的转换（节点名 → 整数下标）也在此提供，供 DataLoader 的
collate_fn 使用。旧版把这两个转换挂在 ``Dataset`` 实例的方法上，
导致加载缓存后的空壳数据集无法直接使用（Train 里被迫重建一个
完整 Dataset 来「借」方法）；重构为模块级函数后彻底解决。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as DateType
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import BASE_FEATURES, MACRO_LOCATIONS

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------

#: 微观节点特征列 = 6 基础 + 6 差分 + 1 格拉汉姆比值 = 13 维
FEATURE_COLUMNS = BASE_FEATURES + [f"{f}_diff" for f in BASE_FEATURES] + ["Graham_Ratio"]

#: 宏观节点数（两条巷道）
NUM_MACRO_NODES = len(MACRO_LOCATIONS)

#: 宏观层边索引：皮顺 ↔ 轨顺（固定，与 MACRO_LOCATIONS 的顺序一致）
MACRO_EDGE_INDEX = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


@dataclass
class DailyGraph:
    """某一天的图快照（预处理阶段）。

    节点特征保存为 Python list（数值），训练时由 collate_fn
    完成归一化并转为 :class:`CollatedDay`。
    """

    date: DateType
    micro_node_ids: List[str]                                   # 微观节点名，如 "皮顺_57.3"
    micro_features: List[List[float]]                           # [num_micro, 13]
    macro_features: List[List[float]]                           # [num_macro, 13]
    micro_edges: List[Tuple[str, str]] = field(default_factory=list)   # 微观层边（节点名）
    macro_edges: List[Tuple[str, str]] = field(default_factory=list)   # 宏观层边（节点名）
    cross_links: List[Tuple[str, str]] = field(default_factory=list)   # 微观节点 → 所属巷道


@dataclass
class CollatedDay:
    """collate_fn 输出的一个时间步：已归一化、已转为设备张量的图数据。"""

    micro_features: torch.Tensor   # [num_micro, 13]
    macro_features: torch.Tensor   # [num_macro, 13]
    micro_edges: torch.Tensor      # [2, E_micro]
    macro_edges: torch.Tensor      # [2, E_macro]（固定：皮顺 ↔ 轨顺）
    cross_links: torch.Tensor      # [2, num_micro]


# ----------------------------------------------------------------------
# 单日图构建
# ----------------------------------------------------------------------

def build_single_day_graph(day_data, date: DateType) -> DailyGraph:
    """构建单日「微观—宏观」两级图。

    Args:
        day_data: 某一天的全部记录（DataFrame）
        date: 该天的日期
    """
    micro_nodes, micro_edges = _build_micro_layer(day_data)
    macro_nodes, macro_edges = _build_macro_layer(micro_nodes)
    cross_links = _build_cross_links(micro_nodes)

    # 节点排序固定，保证特征矩阵行序稳定（目标展平依赖此顺序）
    sorted_micro_keys = sorted(micro_nodes.keys())

    return DailyGraph(
        date=date,
        micro_node_ids=sorted_micro_keys,
        micro_features=[micro_nodes[k] for k in sorted_micro_keys],
        macro_features=[macro_nodes[loc] for loc in MACRO_LOCATIONS],
        micro_edges=micro_edges,
        macro_edges=macro_edges,
        cross_links=cross_links,
    )


def _build_micro_layer(day_data) -> Tuple[Dict[str, List[float]], List[Tuple[str, str]]]:
    """构建微观层：每个测点一个节点，同巷道内节点全连接。

    唯一节点 ID 形如 ``"皮顺_57.3"``（巷道 + 埋深）。
    """
    micro_nodes: Dict[str, List[float]] = {}

    for _, row in day_data.iterrows():
        node_id = f"{row['地点名称']}_{row['埋深']}"
        micro_nodes[node_id] = [float(row.get(col, 0.0)) for col in FEATURE_COLUMNS]

    # 同巷道内全连接（模拟巷道内的空气流通），双向、无自环
    edges: List[Tuple[str, str]] = []
    loc_map: Dict[str, List[str]] = {}
    for nid in sorted(micro_nodes.keys()):
        loc_map.setdefault(nid.split("_")[0], []).append(nid)

    for nodes in loc_map.values():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j]))
                edges.append((nodes[j], nodes[i]))

    return micro_nodes, edges


def _build_macro_layer(micro_nodes: Dict[str, List[float]]) -> Tuple[Dict[str, List[float]], List[Tuple[str, str]]]:
    """构建宏观层：巷道节点 = 所属微观节点特征均值；皮顺 ↔ 轨顺互连。"""
    macro_nodes: Dict[str, List[float]] = {}

    for loc in MACRO_LOCATIONS:
        feats = [v for k, v in micro_nodes.items() if k.startswith(loc)]
        if feats:
            macro_nodes[loc] = np.mean(feats, axis=0).tolist()
        else:  # 数据缺失时兜底为全 0
            macro_nodes[loc] = [0.0] * len(FEATURE_COLUMNS)

    macro_edges = [("皮顺", "轨顺"), ("轨顺", "皮顺")]
    return macro_nodes, macro_edges


def _build_cross_links(micro_nodes: Dict[str, List[float]]) -> List[Tuple[str, str]]:
    """微观 → 宏观的隶属关系：每个微观节点连到其所在巷道的宏观节点。"""
    return [(nid, nid.split("_")[0]) for nid in micro_nodes.keys()]


# ----------------------------------------------------------------------
# 边索引转换（collate_fn 使用）
# ----------------------------------------------------------------------

def convert_edges_to_indices(edges: List[Tuple[str, str]], sorted_node_ids: List[str]) -> torch.Tensor:
    """把「节点名边」转换为 ``[2, num_edges]`` 的整数边索引张量。"""
    node_map = {n: i for i, n in enumerate(sorted_node_ids)}
    indices = [[node_map[u], node_map[v]] for u, v in edges if u in node_map and v in node_map]
    if not indices:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long).t()


def convert_cross_links_to_indices(
    links: List[Tuple[str, str]],
    micro_node_ids: List[str],
) -> torch.Tensor:
    """把跨层连边转换为 ``[2, num_links]`` 的整数索引张量。

    第一行为微观节点下标（按 ``micro_node_ids`` 顺序），
    第二行为宏观节点下标（按 ``MACRO_LOCATIONS`` 顺序）。
    """
    micro_map = {n: i for i, n in enumerate(micro_node_ids)}
    macro_map = {loc: i for i, loc in enumerate(MACRO_LOCATIONS)}
    indices = [[micro_map[u], macro_map[v]] for u, v in links if u in micro_map and v in macro_map]
    if not indices:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long).t()
