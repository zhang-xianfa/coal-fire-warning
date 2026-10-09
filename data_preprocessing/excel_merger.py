# -*- coding: utf-8 -*-
"""
月度束管监测 Excel 合并
=======================

矿井束管监测数据按月存放（``6301采空区束管{月份}月.xls``），每月一个文件、
每天一个工作表。本模块负责把 1-12 月的全部表单合并为一张总表。

原始表单格式（不规范，需专门解析）：

- 第 1 行：列名（如 氧气、一氧化碳、序号…）
- 第 2 行：单位（如 %、ppm…）
- 第 3 行：分隔/空行
- 第 4 行起：数据行，以「序号」列为有效行的标志
"""

from __future__ import annotations

import os
import re

import pandas as pd


class EmptySheetError(Exception):
    """表单为空（部分月份文件里存在空的工作表）。"""


def integrate_monthly_data(
    folder_path: str,
    output_path: str,
    file_pattern: str = "6301采空区束管{month}月.xls",
) -> pd.DataFrame:
    """合并 1-12 月的全部数据文件。

    Args:
        folder_path: 存放月度 .xls 文件的目录
        output_path: 合并结果的输出路径（.xlsx）
        file_pattern: 文件名模板，``{month}`` 会被替换为月份数字

    Returns:
        合并后的 DataFrame
    """
    combined = pd.DataFrame()

    for month in range(1, 13):
        file_path = os.path.join(folder_path, file_pattern.format(month=month))
        try:
            excel_file = pd.ExcelFile(file_path, engine="xlrd")
        except FileNotFoundError:
            print(f"{file_path} 文件未找到，跳过")
            continue
        except Exception as exc:
            print(f"处理 {file_path} 时出错：{exc}")
            continue

        print(f"{month}月有 {len(excel_file.sheet_names)} 个表单")
        for sheet_name in excel_file.sheet_names:
            try:
                sheet_data = read_sheet(excel_file, sheet_name)
            except EmptySheetError as exc:
                print(f"ℹ️ {exc}")
                continue
            except Exception as exc:
                print(f"❌ 处理 {sheet_name} 表单时出错：{exc}")
                continue

            # 去掉整列为空的列后并入总表
            sheet_data = sheet_data.dropna(axis=1, how="all")
            combined = pd.concat([combined, sheet_data], axis=0, ignore_index=True)
            print(f"  ✅ {sheet_name} 处理完成，新增 {len(sheet_data)} 行有效数据")

    if len(combined) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_excel(output_path, index=False, engine="openpyxl")
        print(f"所有数据处理完成，共合并 {len(combined)} 行数据，已保存至 {output_path}")
    else:
        print("未找到有效数据，未生成输出文件")
    return combined


def read_sheet(excel_file: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """解析单个工作表（一天的数据），返回有效数据行。

    Raises:
        EmptySheetError: 表单为空
    """
    # 空表判断：无行 / 无列 / 全部为空值
    sample = excel_file.parse(sheet_name=sheet_name, nrows=1, header=None)
    if sample.empty or sample.shape[1] == 0 or sample.isna().all().all():
        raise EmptySheetError(f"{sheet_name} 表单为空或无有效数据")

    # 前两行是列名与单位，组合成 "名称(单位)" 形式的列名
    headers = excel_file.parse(sheet_name=sheet_name, nrows=2)
    column_names = headers.iloc[0].tolist()
    column_units = headers.iloc[1].tolist()
    new_columns = [
        _combine_name_unit(name, unit, idx)
        for idx, (name, unit) in enumerate(zip(column_names, column_units))
    ]

    # 数据从第 4 行开始
    data_df = excel_file.parse(sheet_name=sheet_name, skiprows=3, header=None)
    data_df = _align_columns(data_df, new_columns)

    # 以「序号」列筛选有效数据行
    if "序号" not in data_df.columns:
        print(f"❌ 该表单无「序号」列，跳过处理")
        return pd.DataFrame()

    valid = data_df[
        data_df["序号"].notna()
        & data_df["序号"].astype(str).str.strip().str.isdigit()
    ].copy()
    return valid


def analyze_nonzero_ratio(file_path: str, sheet_name: str = "Sheet1") -> pd.DataFrame | None:
    """统计合并后数据中各数值列的非零值数量与比例（数据质量检查）。

    Args:
        file_path: 合并后的 .xlsx 文件
        sheet_name: 工作表名

    Returns:
        包含「有效数值总数 / 非零数量 / 非零比例%」的 DataFrame，失败返回 None
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
        print(f"✅ 成功读取文件，共 {df.shape[0]} 行数据，{df.shape[1]} 列数据")
    except Exception as exc:
        print(f"❌ 读取文件出错：{exc}")
        return None

    # 所有列统一清洗为数值（字符串列会剥离非数值字符后转换）
    df_numeric = df.apply(_to_numeric_column)

    # 丢弃全为 NaN 的列
    valid_cols = [c for c in df_numeric.columns if df_numeric[c].notna().any()]
    if not valid_cols:
        print("❌ 未找到有效数值列，无法统计非零值")
        return None

    df_valid = df_numeric[valid_cols]
    non_zero = df_valid.astype(bool).sum(axis=0)
    total = df_valid["序号"].notna().sum() if "序号" in df_valid else len(df_valid)
    ratio = (non_zero / total * 100).round(2)

    result = pd.DataFrame({
        "有效数值总数": total,
        "非零数量": non_zero,
        "非零比例%": ratio,
    })
    print(f"\n📈 非零值统计完成（有效数值列 {len(valid_cols)} 列）")
    return result


# ----------------------------------------------------------------------
# 内部工具函数
# ----------------------------------------------------------------------

def _combine_name_unit(name, unit, idx: int) -> str:
    """把列名与单位组合为 "名称(单位)"。"""
    name = str(name).strip() if pd.notna(name) else ""
    unit = str(unit).strip() if pd.notna(unit) else ""
    if name and unit:
        return f"{name}({unit})"
    if name:
        return name
    return f"Unnamed_{idx}"


def _align_columns(data_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """处理列名数量与数据列数不匹配的情况。"""
    data_cols = data_df.shape[1]
    if len(columns) > data_cols:
        columns = columns[:data_cols]
    elif len(columns) < data_cols:
        columns += [f"Unnamed_{i}" for i in range(len(columns), data_cols)]
    data_df.columns = columns
    return data_df


def _to_numeric_column(series: pd.Series) -> pd.Series:
    """把一列数据清洗为纯数值（非数值内容置为 NaN）。"""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = series.astype(str).apply(
        lambda x: re.sub(r"[^\d\-.eE]", "", x.strip())
    ).replace("", "NaN")
    return pd.to_numeric(cleaned, errors="coerce")
