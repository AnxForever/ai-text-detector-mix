"""
BERT AI文本检测模型训练脚本 - 改进版
集成所有长度平衡改进策略
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 导入自定义模块
from scripts.bert_prep.create_bert_dataset import AIDetectionDataset, dynamic_collate_fn
from scripts.training.length_weighted_loss import LengthWeightedLoss

class BERTTrainer:
    def __init__(
        self,
        model_name='bert-base-chinese',
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5,
        warmup_steps=500,
        weight_decay=0.01,
        use_length_weighted_loss=True,
        loss_alpha=0.3,
        device=None,
        output_dir='models/bert_improved'
    ):
        """
        初始化训练器

        参数:
            model_name: BERT模型名称
            max_length: 最大序列长度
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            warmup_steps: 预热步数
            weight_decay: 权重衰减
            use_length_weighted_loss: 是否使用长度加权损失
            loss_alpha: 长度加权损失的alpha参数
            device: 训练设备
            output_dir: 输出目录
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_length_weighted_loss = use_length_weighted_loss
        self.loss_alpha = loss_alpha
        self.output_dir = output_dir

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化模型和tokenizer
        self._init_model()

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

    def _init_model(self):
        """初始化模型和tokenizer"""
        print("\n正在加载BERT模型和tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)
        print(f"✓ 模型加载成功: {self.model_name}")

    def load_data(self, train_csv, val_csv, test_csv):
        """加载数据集"""
        print("\n正在加载数据集...")

        # 读取CSV
        train_df = pd.read_csv(train_csv, encoding='utf-8-sig')
        val_df = pd.read_csv(val_csv, encoding='utf-8-sig')
        test_df = pd.read_csv(test_csv, encoding='utf-8-sig')

        print(f"  训练集: {len(train_df)} 条")
        print(f"  验证集: {len(val_df)} 条")
        print(f"  测试集: {len(test_df)} 条")

        # 创建Dataset
        self.train_dataset = AIDetectionDataset(
            train_df['text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            max_length=self.max_length,
            use_dynamic_padding=True
        )

        self.val_dataset = AIDetectionDataset(
            val_df['text'].tolist(),
            val_df['label'].tolist(),
            self.tokenizer,
            max_length=self.max_length,
            use_dynamic_padding=True
        )

        self.test_dataset = AIDetectionDataset(
            test_df['text'].tolist(),
            test_df['label'].tolist(),
            self.tokenizer,
            max_length=self.max_length,
            use_dynamic_padding=True
        )

        # 创建DataLoader（使用动态padding）
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dynamic_collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dynamic_collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=dynamic_collate_fn
        )

        print("✓ 数据加载完成")
        print(f"✓ 使用动态Padding策略（batch级别自适应）")

    def _setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # 损失函数
        if self.use_length_weighted_loss:
            self.criterion = LengthWeightedLoss(
                alpha=self.loss_alpha,
                max_length=self.max_length,
                reduction='mean'
            )
            print(f"✓ 使用长度加权损失 (alpha={self.loss_alpha})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("✓ 使用标准交叉熵损失")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='训练中')
        for batch in pbar:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits

            # 计算损失
            if self.use_length_weighted_loss:
                # 计算实际长度
                lengths = attention_mask.sum(dim=1)
                loss = self.criterion(logits, labels, lengths=lengths, attention_masks=attention_mask)
            else:
                loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='评估中'):
                # 移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                logits = outputs.logits

                # 计算损失
                if self.use_length_weighted_loss:
                    lengths = attention_mask.sum(dim=1)
                    loss = self.criterion(logits, labels, lengths=lengths, attention_masks=attention_mask)
                else:
                    loss = self.criterion(logits, labels)

                # 统计
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return avg_loss, accuracy, precision, recall, f1

    def train(self):
        """完整训练流程"""
        print("\n" + "="*70)
        print("开始训练")
        print("="*70)

        # 设置优化器
        self._setup_optimizer()

        best_val_f1 = 0

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(self.val_loader)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            # 打印结果
            print(f"\n训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            print(f"       Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_model('best_model')
                print(f"✓ 保存最佳模型 (F1={val_f1:.4f})")

        # 保存最终模型
        self.save_model('final_model')
        print("\n✓ 训练完成！")

    def test(self):
        """在测试集上评估"""
        print("\n" + "="*70)
        print("测试集评估")
        print("="*70)

        # 加载最佳模型
        self.load_model('best_model')

        # 评估
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.evaluate(self.test_loader)

        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")

        # 详细分类报告
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels']

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        print("\n分类报告:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=['人类文本', 'AI文本'],
            digits=4
        ))

        # 保存测试结果
        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 测试结果已保存: {os.path.join(self.output_dir, 'test_results.json')}")

        return test_results

    def save_model(self, name='model'):
        """保存模型"""
        save_path = os.path.join(self.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        # 保存模型和tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 保存训练历史
        with open(os.path.join(save_path, 'training_history.json'), 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        # 保存配置
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'use_length_weighted_loss': self.use_length_weighted_loss,
            'loss_alpha': self.loss_alpha
        }

        with open(os.path.join(save_path, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_model(self, name='model'):
        """加载模型"""
        load_path = os.path.join(self.output_dir, name)

        self.model = BertForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(load_path)

        print(f"✓ 模型已加载: {load_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train BERT AI text detector")
    parser.add_argument("--model-name", default="bert-base-chinese",
                        help="Model name or path (e.g., models/bert_improved/best_model)")
    parser.add_argument("--train-csv", default="datasets/bert_v2/train.csv", help="Train CSV path")
    parser.add_argument("--val-csv", default="datasets/bert_v2/val.csv", help="Val CSV path")
    parser.add_argument("--test-csv", default="datasets/bert_v2/test.csv", help="Test CSV path")
    parser.add_argument("--output-dir", default="models/bert_improved", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--loss-alpha", type=float, default=0.3, help="Length-weighted loss alpha")
    parser.add_argument("--no-length-weighted-loss", action="store_true",
                        help="Disable length-weighted loss")
    args = parser.parse_args()

    print("="*70)
    print("BERT AI文本检测模型训练 - 改进版")
    print("="*70)
    print("\n集成改进策略:")
    print("  ✓ 长度分层数据集（完美平衡）")
    print("  ✓ 动态Padding（减少长度信号泄露）")
    print("  ✓ 长度加权损失（提升短文本性能）")
    print("  ✓ 智能优化器配置")

    # 创建训练器
    trainer = BERTTrainer(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_length_weighted_loss=not args.no_length_weighted_loss,
        loss_alpha=args.loss_alpha,
        output_dir=args.output_dir
    )

    # 加载数据
    trainer.load_data(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv
    )

    # 训练
    trainer.train()

    # 测试
    trainer.test()

    print("\n" + "="*70)
    print("✅ 所有流程完成！")
    print("="*70)
    print(f"\n模型保存位置: {trainer.output_dir}")
    print("\n下一步:")
    print("  1. 运行分长度区间评估:")
    print("     python scripts/evaluation/length_aware_evaluation.py \\")
    print(f"       --model_path {trainer.output_dir}/best_model \\")
    print(f"       --test_csv {args.test_csv}")
    print("\n  2. 查看训练历史:")
    print("     cat models/bert_improved/best_model/training_history.json")


if __name__ == '__main__':
    main()
