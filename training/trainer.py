# -*- coding: utf-8 -*-
"""
训练流程
========

数据准备 → 训练循环 → 测试评估：

- **数据准备**：读取缓存（存在时）或重建数据集；**按时间顺序**划分
  训练/验证/测试（前 70% / 中 15% / 后 15%，避免随机划分造成的时序泄漏）；
  只在训练集上拟合标准化参数；
- **训练循环**：Adam（lr=5e-4, weight_decay=1e-2 抑制过拟合）+
  风险感知损失；按验证损失保存最优模型；
- **测试评估**：加载最优模型（而非最后一个 epoch），还原真实量纲后
  施加物理约束（浓度非负、O2 ≤ 22%），输出全局/分气体的 MSE/MAE/R2，
  并落盘 ``output/metrics.csv`` 与对比图。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Subset

from config import BEST_MODEL_PATH, OUTPUT_DIR, PROCESSED_DATA_CACHE, TRAIN_DATA_FILE
from dataset.coal_dataset import CoalMineDataset
from dataset.graph_builder import (
    MACRO_EDGE_INDEX,
    CollatedDay,
    DailyGraph,
    convert_cross_links_to_indices,
    convert_edges_to_indices,
)
from models.hierarchical_gnn import ModelManager
from training.losses import RiskAwareLoss
from training.normalizer import Normalizer


class Trainer:
    """训练流程编排。"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = Normalizer()

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        # 缓存命中时跳过 Excel 读取与预处理
        if PROCESSED_DATA_CACHE.exists():
            print("加载缓存数据...")
            self.dataset = CoalMineDataset.load_processed_data(str(PROCESSED_DATA_CACHE))
        else:
            self.dataset = CoalMineDataset(str(TRAIN_DATA_FILE), self.config["seq_len"])
            self.dataset.save_processed_data(str(PROCESSED_DATA_CACHE))

        # 训练 70% / 验证 15% / 测试 15%
        # 默认**按时间顺序**切分：序列本身按日期升序排列，用前段训练、后段测试才符合
        # "预测未来"的设定。random 模式只用于复现旧口径、做泄漏对照（结果见 README）。
        total = len(self.dataset)
        train_size = int(self.config["train_ratio"] * total)
        val_size = int(self.config["val_ratio"] * total)
        test_size = total - train_size - val_size
        self.split_mode = self.config.get("split_mode", "time")

        if self.split_mode == "random":
            perm = torch.randperm(
                total, generator=torch.Generator().manual_seed(self.config["seed"])
            ).tolist()
            train_idx = perm[:train_size]
            val_idx = perm[train_size:train_size + val_size]
            test_idx = perm[train_size + val_size:]
        else:
            train_idx = list(range(0, train_size))
            val_idx = list(range(train_size, train_size + val_size))
            test_idx = list(range(train_size + val_size, total))
        print(f"数据划分：{self.split_mode}（训练 {len(train_idx)} / 验证 {len(val_idx)} / 测试 {len(test_idx)}）")

        self.train_ds = Subset(self.dataset, train_idx)
        self.val_ds = Subset(self.dataset, val_idx)
        self.test_ds = Subset(self.dataset, test_idx)

        # 只在训练集上拟合标准化参数
        self.normalizer.fit(self.dataset, train_idx)

    # ------------------------------------------------------------------
    # collate：DailyGraph → CollatedDay（归一化 + 张量化 + 上设备）
    # ------------------------------------------------------------------

    def _collate_day(self, day: DailyGraph) -> CollatedDay:
        micro, macro = self.normalizer.transform_day(day)
        return CollatedDay(
            micro_features=micro.to(self.device),
            macro_features=macro.to(self.device),
            micro_edges=convert_edges_to_indices(day.micro_edges, day.micro_node_ids).to(self.device),
            macro_edges=MACRO_EDGE_INDEX.to(self.device),
            cross_links=convert_cross_links_to_indices(day.cross_links, day.micro_node_ids).to(self.device),
        )

    def _collate_fn(self, batch):
        seqs, tgts = zip(*batch)
        norm_seqs = [[self._collate_day(day) for day in seq] for seq in seqs]
        norm_tgts = torch.stack([self.normalizer.transform_target(t) for t in tgts]).to(self.device)
        return norm_seqs, norm_tgts

    # ------------------------------------------------------------------
    # 训练与评估
    # ------------------------------------------------------------------

    def run(self) -> None:
        # 固定随机种子，保证结果可复现
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])

        self.prepare_data()

        train_loader = DataLoader(
            self.train_ds, batch_size=self.config["batch_size"],
            shuffle=True, collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.config["batch_size"], collate_fn=self._collate_fn,
        )
        test_loader = DataLoader(
            self.test_ds, batch_size=1, collate_fn=self._collate_fn,
        )

        model, model_config = ModelManager.create_model_from_dataset(
            self.dataset, self.config["hidden_dim"]
        )
        model = model.to(self.device)

        # weight_decay 即 L2 正则化，抑制过拟合
        optimizer = optim.Adam(
            model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"]
        )
        criterion = RiskAwareLoss(self.device, penalty_factor=2.0)
        print("Trainer: 使用风险感知损失函数 (Penalty=2.0, O2_Weight=15.0)...")

        print(
            f"\n模型结构:\n"
            f"  输入: 微观节点({model_config['num_micro_nodes']}) + 宏观节点({model_config['num_macro_nodes']})"
        )
        print("开始训练...")

        best_val_loss = float("inf")

        for epoch in range(self.config["epochs"]):
            # ---- 训练 ----
            model.train()
            train_loss = 0.0
            for seqs, tgts in train_loader:
                optimizer.zero_grad()
                preds = model(seqs)
                loss = criterion(preds, tgts)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # ---- 验证 ----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for seqs, tgts in val_loader:
                    preds = model(seqs)
                    val_loss += criterion(preds, tgts).item()

            avg_t = train_loss / len(train_loader)
            avg_v = val_loss / len(val_loader)

            # 每 5 轮打印一次日志，或有提升时打印
            if (epoch + 1) % 5 == 0 or avg_v < best_val_loss:
                print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

            # 按验证损失保存最优模型
            if avg_v < best_val_loss:
                best_val_loss = avg_v
                ModelManager.save_model(model, str(BEST_MODEL_PATH), model_config)

        self.evaluate(test_loader)

    def evaluate(self, loader) -> None:
        """加载最优模型（Best Val Loss），在测试集上评估。"""
        print("\n" + "=" * 40)
        print("正在测试最佳模型 (Best Val Loss Model)")
        print("=" * 40)

        # 务必加载表现最好的那个模型，而不是最后一个 Epoch 的模型
        model = ModelManager.load_model(str(BEST_MODEL_PATH), self.device)
        model.eval()

        all_preds, all_tgts = [], []

        with torch.no_grad():
            for seqs, tgts in loader:
                preds = model(seqs)
                pred_real = self.normalizer.inverse_target(preds)
                tgt_real = self.normalizer.inverse_target(tgts)

                # 物理约束：浓度非负；O2 不超过环境上限 22%
                pred_real[pred_real < 0] = 0
                o2_indices = list(range(0, pred_real.shape[1], 3))
                pred_real[:, o2_indices] = np.clip(pred_real[:, o2_indices], 0, 22.0)

                all_preds.append(pred_real)
                all_tgts.append(tgt_real)

        all_preds = np.vstack(all_preds)
        all_tgts = np.vstack(all_tgts)

        metric_rows = []

        def _metrics(p, t) -> Dict[str, float]:
            """一组预测的结果：MSE / MAE / R2（决定系数）。"""
            mse = float(np.mean((p - t) ** 2))
            mae = float(np.mean(np.abs(p - t)))
            ss_res = float(np.sum((t - p) ** 2))
            ss_tot = float(np.sum((t - t.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
            return {"mse": mse, "mae": mae, "r2": r2}

        # 1. 全局指标
        g = _metrics(all_preds, all_tgts)
        print(f"全局 MSE: {g['mse']:.4f}")
        print(f"全局 MAE: {g['mae']:.4f}")
        print(f"全局 R2 : {g['r2']:.4f}")
        print("-" * 40)
        metric_rows.append({"范围": "全局", "气体": "全部",
                            "MSE": f"{g['mse']:.4f}", "MAE": f"{g['mae']:.4f}",
                            "R2": f"{g['r2']:.4f}"})

        # 2. 分气体指标
        gas_names = ["氧气 (O2)", "一氧化碳 (CO)", "二氧化碳 (CO2)"]
        for i, gas_name in enumerate(gas_names):
            col_indices = list(range(i, all_preds.shape[1], 3))
            p_gas = all_preds[:, col_indices]
            t_gas = all_tgts[:, col_indices]

            m = _metrics(p_gas, t_gas)
            print(f"{gas_name} 预测结果:")
            print(f"   MSE: {m['mse']:.4f}")
            print(f"   MAE: {m['mae']:.4f}")
            print(f"   R2 : {m['r2']:.4f}")
            print("-" * 20)
            metric_rows.append({"范围": "分气体", "气体": gas_name,
                                "MSE": f"{m['mse']:.4f}", "MAE": f"{m['mae']:.4f}",
                                "R2": f"{m['r2']:.4f}"})

        # 指标落盘，供论文/简历引用时溯源：
        # output/ 是本次运行产物，results/ 随仓库提交便于他人复核（按划分方式分文件）
        metrics_paths = [
            OUTPUT_DIR / "metrics.csv",
            Path(__file__).resolve().parents[1] / "results"
            / f"metrics_{getattr(self, 'split_mode', 'time')}.csv",
        ]
        for metrics_path in metrics_paths:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["范围", "气体", "MSE", "MAE", "R2"])
                writer.writeheader()
                writer.writerows(metric_rows)
            print(f"\n指标已保存至: {metrics_path}")

        # 3. 抽样查看（前 2 个样本的所有节点）
        print("\n预测值 vs 真实值 (前2个样本 - 所有节点):")
        num_nodes = all_preds.shape[1] // 3
        for sample_idx in range(min(2, len(all_preds))):
            print(f"\n>>>>> 测试样本 {sample_idx + 1} <<<<<")
            print(f"{'节点ID':<8} | {'气体':<4} | {'真实值':>8} | {'预测值':>8} | {'误差':>8}")
            print("-" * 50)
            for node_i in range(num_nodes):
                base = node_i * 3
                # O2
                r_o2, p_o2 = all_tgts[sample_idx, base], all_preds[sample_idx, base]
                print(f"Node {node_i + 1:<3} | O2   | {r_o2:8.2f} | {p_o2:8.2f} | {p_o2 - r_o2:8.2f}")
                # CO
                r_co, p_co = all_tgts[sample_idx, base + 1], all_preds[sample_idx, base + 1]
                print(f"         | CO   | {r_co:8.2f} | {p_co:8.2f} | {p_co - r_co:8.2f}")
            print("-" * 50)

        # 4. 绘图（节点 1 的三种气体预测对比）
        plt.figure(figsize=(18, 6))
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False

        for i, gas_name in enumerate(gas_names):
            p_series = all_preds[:, i]
            t_series = all_tgts[:, i]
            ax = plt.subplot(1, 3, i + 1)
            ax.plot(t_series, "o-", label="真实值", alpha=0.6, markersize=6)
            ax.plot(p_series, "x--", label="预测值", alpha=0.8, markersize=6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"节点1 - {gas_name}")
            ax.set_xlabel("测试样本编号")
            ax.set_ylabel("浓度值")
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle="--")

        save_path = OUTPUT_DIR / "prediction_result_detailed.png"
        plt.savefig(save_path)
        print(f"\n可视化散点图已保存至: {save_path}")
