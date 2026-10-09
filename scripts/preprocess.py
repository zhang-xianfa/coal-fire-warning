# -*- coding: utf-8 -*-
"""
一键预处理：合并月度报表 → 数据清洗 → 训练数据
================================================

用法::

    python scripts/preprocess.py

流程：

1. 若 ``data/raw/6301/`` 下存在月度 .xls 报表，合并为总表并做数据质量检查；
2. 清洗总表（按天过滤 + 删除回风数据），产出 ``data/6301处理.xlsx``。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data_preprocessing.cleaner import clean_for_training
from data_preprocessing.excel_merger import analyze_nonzero_ratio, integrate_monthly_data


def main() -> None:
    # 1. 合并月度报表
    raw_dir = config.RAW_MONTHLY_DIR
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print("=" * 60)
        print("步骤 1/2：合并 1-12 月月度束管报表")
        print("=" * 60)
        integrate_monthly_data(str(raw_dir), str(config.MERGED_DATA_FILE), config.RAW_FILE_PATTERN)
        print()
        analyze_nonzero_ratio(str(config.MERGED_DATA_FILE))
    else:
        print(f"ℹ️ 未找到月度原始报表目录 {raw_dir}，跳过合并步骤")

    # 2. 清洗
    print("\n" + "=" * 60)
    print("步骤 2/2：数据清洗")
    print("=" * 60)
    if config.MERGED_DATA_FILE.exists():
        clean_for_training(str(config.MERGED_DATA_FILE), str(config.TRAIN_DATA_FILE))
    elif config.TRAIN_DATA_FILE.exists():
        print(f"ℹ️ 合并总表不存在，但训练数据 {config.TRAIN_DATA_FILE} 已存在，无需清洗")
    else:
        raise SystemExit("既没有月度报表也没有训练数据文件，请检查 data/ 目录")


if __name__ == "__main__":
    main()
