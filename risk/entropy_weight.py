# -*- coding: utf-8 -*-
"""
熵权法
======

信息熵越小 → 该特征的数据差异越大 → 应获得更高权重。
"""

from __future__ import annotations

import numpy as np


def entropy_weights(data_matrix: np.ndarray) -> np.ndarray:
    """传统熵权法计算特征权重（所有样本权重相等）。

    Args:
        data_matrix: 特征矩阵，形状 (样本数, 特征数)

    Returns:
        各特征权重（和为 1）
    """
    n_features = data_matrix.shape[1]

    # 1. Min-Max 归一化到 [0, 1]
    min_vals = data_matrix.min(axis=0)
    max_vals = data_matrix.max(axis=0)
    normalized = (data_matrix - min_vals) / (max_vals - min_vals + 1e-8)

    n = len(data_matrix)
    if n <= 1:  # 样本数不足，权重取平均
        return np.ones(n_features) / n_features

    # 2. 概率分布（所有样本权重相等）
    p = normalized / (normalized.sum(axis=0) + 1e-8)

    # 3. 逐特征计算信息熵（p == 0 时贡献为 0，+1e-8 防止 log(0)）
    k = 1.0 / np.log(n)  # 熵系数：样本数固定时使熵落在 [0, 1]
    e = -k * (p * np.log(p + 1e-8)).sum(axis=0)

    # 4. 差异系数 → 归一化权重
    d = 1 - e
    if d.sum() == 0:
        return np.ones(n_features) / n_features
    return d / (d.sum() + 1e-8)
