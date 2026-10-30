# -*- coding: utf-8 -*-
"""
评估入口（不重新训练）
======================

用法::

    python scripts/evaluate.py

加载 ``output/best_model.pth``，在**按时间顺序划分出的测试集**上重算
MSE / MAE / R²，并写出 ``output/metrics.csv`` 与 ``results/metrics.csv``。
适合改完评估逻辑后快速复算指标，不必再等一轮完整训练。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from config import TRAIN_CONFIG
from training.trainer import Trainer


def main() -> None:
    trainer = Trainer(TRAIN_CONFIG)
    trainer.prepare_data()  # 重建数据集、按时间划分、只在训练集拟合归一化参数

    test_loader = DataLoader(
        trainer.test_ds, batch_size=1, collate_fn=trainer._collate_fn,
    )
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
