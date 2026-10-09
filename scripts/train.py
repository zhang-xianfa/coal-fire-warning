# -*- coding: utf-8 -*-
"""
训练入口
========

用法::

    python scripts/train.py

流程：读取缓存（或重建数据集）→ 70/15/15 划分 → 训练 → 测试评估。
最优模型保存至 ``output/best_model.pth``，评估对比图保存至 ``output/``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TRAIN_CONFIG
from training.trainer import Trainer

if __name__ == "__main__":
    Trainer(TRAIN_CONFIG).run()
