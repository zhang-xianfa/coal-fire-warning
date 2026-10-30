# -*- coding: utf-8 -*-
"""
训练入口
========

用法::

    python scripts/train.py                     # 默认按时间顺序切分
    python scripts/train.py --split-mode random # 随机切分（仅用于泄漏对照）

流程：读取缓存（或重建数据集）→ 70/15/15 划分 → 训练 → 测试评估。
最优模型保存至 ``output/best_model.pth``，评估指标保存至 ``output/metrics.csv``
与 ``results/metrics_<划分方式>.csv``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TRAIN_CONFIG
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="训练分层时空 GNN 模型")
    parser.add_argument(
        "--split-mode", choices=["time", "random"],
        default=TRAIN_CONFIG.get("split_mode", "time"),
        help="数据划分方式：time（按时间，默认）/ random（随机，做泄漏对照用）",
    )
    args = parser.parse_args()

    config = dict(TRAIN_CONFIG, split_mode=args.split_mode)
    Trainer(config).run()


if __name__ == "__main__":
    main()
