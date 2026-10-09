# -*- coding: utf-8 -*-
"""
风险分析入口（熵权法评分 + 热力图）
====================================

用法::

    python scripts/evaluate_risk.py

流程：

1. 加载训练数据，整理为 7 个监测点的逐日序列；
2. 统计 CO 超过 24ppm 预警阈值的情况；
3. 熵权法 + 趋势修正 + CO 压制，计算逐日风险评分；
4. 生成单日热力图 / 3D 巷道图 / 时序总览 / 30 天详图，
   输出至 ``output/risk_heatmaps/``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from risk.heatmap import HeatmapGenerator
from risk.scorer import RiskScorer


def main() -> None:
    scorer = RiskScorer(str(config.TRAIN_DATA_FILE))
    scorer.prepare_daily_data()
    scorer.stat_co_exceed()

    results = scorer.calculate_risk_scores()
    if not results:
        return

    output_dir = config.OUTPUT_DIR / "risk_heatmaps"
    HeatmapGenerator(results, str(output_dir)).run_all()


if __name__ == "__main__":
    main()
