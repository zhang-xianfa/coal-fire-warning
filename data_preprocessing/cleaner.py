# -*- coding: utf-8 -*-
"""
数据清洗
========

对合并后的总表做训练前的清洗：

1. 从「化验日期」中提取日期部分；
2. 只保留每天记录数 ≥ 7 的日期（保证当天的 6 个测点 + 回风数据齐全）；
3. 删除「回风」地点的数据（模型只建模皮顺 / 轨顺两条巷道的测点，
   回风数据由风险可视化模块单独使用）。
"""

from __future__ import annotations

import pandas as pd

#: 每天最少需要的记录条数（保证测点数据完整）
MIN_RECORDS_PER_DAY = 7

#: 需要剔除的地点（回风数据不参与图建模）
EXCLUDED_LOCATION = "回风"


def clean_for_training(input_file: str, output_file: str) -> pd.DataFrame:
    """清洗总表数据并保存。

    Args:
        input_file: 合并后的总表路径（.xlsx）
        output_file: 清洗结果的保存路径

    Returns:
        清洗后的 DataFrame
    """
    df = pd.read_excel(input_file)

    # 1. 提取日期部分（去掉时间）
    df["日期"] = df["化验日期"].astype(str).str.split().str[0]

    # 2. 只保留记录数充足的日期
    daily_counts = df["日期"].value_counts()
    valid_dates = daily_counts[daily_counts >= MIN_RECORDS_PER_DAY].index
    df_filtered = df[df["日期"].isin(valid_dates)].copy()

    print(f"原始数据总条数：{len(df)}")
    print(f"有效天数：{len(valid_dates)}")
    print(f"筛选后数据总条数：{len(df_filtered)}")

    # 3. 删除回风数据
    df_final = df_filtered[df_filtered["地点名称"] != EXCLUDED_LOCATION].copy()
    print(f"删除回风数据后总条数：{len(df_final)}")

    # 验证：每天应恰好剩余 6 条（皮顺 3 个深度 + 轨顺 3 个深度）
    final_counts = df_final["日期"].value_counts()
    print("\n最终每天数据条数统计：")
    print(final_counts.value_counts().sort_index())

    # 保存（去掉临时日期列）
    df_final.drop(columns=["日期"]).to_excel(output_file, index=False)
    print(f"\n处理完成！最终数据已保存为 {output_file}（{len(df_final)} 条）")

    # 数据分布信息
    print("\n地点分布：")
    print(df_final["地点名称"].value_counts())
    print(f"\n日期范围：{df_final['化验日期'].min()} 到 {df_final['化验日期'].max()}")
    return df_final
