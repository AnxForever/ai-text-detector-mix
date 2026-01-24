"""
测试AI生成文本的检测脚本
专门处理包含markdown格式的文本

使用方法:
1. 直接运行: python test_ai_generated_text.py
2. 测试单个文本: python test_ai_generated_text.py --text "要测试的文本"
3. 批量测试: python test_ai_generated_text.py --batch
"""

import sys
import io
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import re
import argparse
from datetime import datetime


class AITextDetector:
    """AI文本检测器"""

    def __init__(self, model_path='models/bert_improved/best_model'):
        """加载模型和tokenizer"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载模型...", flush=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  设备: {self.device}", flush=True)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"  ✓ 模型加载成功", flush=True)

    def analyze_text_format(self, text):
        """分析文本中的格式特征"""
        features = {
            'has_bold': bool(re.search(r'\*\*[^*]+\*\*', text)),
            'has_italic': bool(re.search(r'_[^_]+_', text)),
            'has_heading': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
            'has_list': bool(re.search(r'^[\-\*]\s', text, re.MULTILINE)),
            'has_divider': bool(re.search(r'^[\-*_]{3,}$', text, re.MULTILINE)),
            'has_code_block': bool(re.search(r'```', text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'line_count': len([l for l in text.split('\n') if l.strip()]),
        }
        return features

    def predict(self, text, show_analysis=True):
        """
        预测文本是AI还是人类生成

        Args:
            text: 输入文本（可包含markdown格式）
            show_analysis: 是否显示格式分析

        Returns:
            dict: 包含预测结果和置信度
        """
        # 1. 格式分析
        if show_analysis:
            print("\n" + "="*60, flush=True)
            print("文本格式分析", flush=True)
            print("="*60, flush=True)

            features = self.analyze_text_format(text)

            print(f"字符数: {len(text)}", flush=True)
            print(f"段落数: {features['paragraph_count']}", flush=True)
            print(f"行数: {features['line_count']}", flush=True)
            print("\nMarkdown特征:", flush=True)
            print(f"  加粗（**）: {'✓' if features['has_bold'] else '✗'}", flush=True)
            print(f"  斜体（_）: {'✓' if features['has_italic'] else '✗'}", flush=True)
            print(f"  标题（#）: {'✓' if features['has_heading'] else '✗'}", flush=True)
            print(f"  列表（-/*）: {'✓' if features['has_list'] else '✗'}", flush=True)
            print(f"  分割线（---）: {'✓' if features['has_divider'] else '✗'}", flush=True)
            print(f"  代码块（```）: {'✓' if features['has_code_block'] else '✗'}", flush=True)

        # 2. Tokenization（markdown符号会被tokenize）
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        # 3. 预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()

        # 4. 结果
        ai_confidence = probs[0][1].item()
        human_confidence = probs[0][0].item()

        result = {
            'prediction': 'AI' if prediction == 1 else '人类',
            'ai_confidence': ai_confidence,
            'human_confidence': human_confidence,
            'token_count': attention_mask.sum().item(),
            'format_features': self.analyze_text_format(text) if show_analysis else None
        }

        return result

    def predict_batch(self, texts, show_summary=True):
        """批量预测"""
        results = []

        print(f"\n批量测试 {len(texts)} 个文本...\n", flush=True)

        for i, text in enumerate(texts, 1):
            print(f"[{i}/{len(texts)}] 测试中...", flush=True)
            result = self.predict(text, show_analysis=False)
            results.append(result)

            # 简要输出
            label = result['prediction']
            conf = result['ai_confidence'] if label == 'AI' else result['human_confidence']
            print(f"  结果: {label} (置信度: {conf:.2%})\n", flush=True)

        # 统计摘要
        if show_summary:
            ai_count = sum(1 for r in results if r['prediction'] == 'AI')
            human_count = len(results) - ai_count
            avg_ai_conf = sum(r['ai_confidence'] for r in results) / len(results)

            print("="*60, flush=True)
            print("批量测试摘要", flush=True)
            print("="*60, flush=True)
            print(f"总样本数: {len(results)}", flush=True)
            print(f"检测为AI: {ai_count} ({ai_count/len(results):.1%})", flush=True)
            print(f"检测为人类: {human_count} ({human_count/len(results):.1%})", flush=True)
            print(f"平均AI置信度: {avg_ai_conf:.2%}", flush=True)

        return results


def test_with_examples():
    """使用示例文本进行测试"""

    # 示例1: 包含多种markdown格式的AI文本
    example1 = """## 人工智能简介

人工智能（Artificial Intelligence, AI）是计算机科学的一个重要分支。

**核心技术领域：**
- 机器学习（Machine Learning）
- 深度学习（Deep Learning）
- 自然语言处理（NLP）
- 计算机视觉（CV）

---

### 发展历程

人工智能经历了三次浪潮：
1. 符号主义时代（1950s-1980s）
2. 连接主义复兴（1980s-2010s）
3. 深度学习革命（2010s至今）

**当前热点：**
- 大语言模型（LLMs）
- 生成式AI
- AGI（通用人工智能）

```python
# 简单的AI应用示例
def classify_text(text):
    model = load_model('bert-base')
    return model.predict(text)
```

总结：AI技术正在快速发展，将深刻影响人类社会。"""

    # 示例2: 纯文本AI输出（无格式）
    example2 = """人工智能是一门研究如何让计算机模拟人类智能的学科。它包括机器学习、深度学习、自然语言处理等多个子领域。近年来，随着算力的提升和数据量的增加，AI技术取得了突破性进展。特别是大语言模型的出现，使得AI在文本理解和生成方面达到了前所未有的水平。未来，AI将在医疗、教育、金融等领域发挥更大作用。"""

    # 示例3: 短文本（测试长度鲁棒性）
    example3 = """**AI定义：** 人工智能是模拟人类智能的技术。"""

    # 示例4: 长文本带复杂格式
    example4 = """# 深度学习完整指南

## 1. 基础概念

深度学习是机器学习的一个子集，使用多层神经网络进行特征学习。

### 1.1 神经网络结构

**基本组成：**
- 输入层（Input Layer）
- 隐藏层（Hidden Layers）
- 输出层（Output Layer）

每层包含多个神经元，通过权重连接。

---

## 2. 核心算法

### 2.1 反向传播

反向传播算法是训练神经网络的核心：

```python
def backpropagation(network, data, labels):
    # 前向传播
    outputs = forward_pass(network, data)

    # 计算损失
    loss = compute_loss(outputs, labels)

    # 反向传播梯度
    gradients = backward_pass(network, loss)

    # 更新权重
    update_weights(network, gradients)

    return loss
```

### 2.2 优化器

常用优化器：
* SGD（随机梯度下降）
* Adam（自适应学习率）
* RMSprop

---

## 3. 应用场景

| 领域 | 应用 | 技术 |
|------|------|------|
| 视觉 | 图像识别 | CNN |
| 语言 | 机器翻译 | Transformer |
| 语音 | 语音识别 | RNN/LSTM |

### 3.1 计算机视觉

**典型任务：**
- 图像分类
- 目标检测
- 语义分割
- 实例分割

卷积神经网络（CNN）是处理图像的主要架构。

### 3.2 自然语言处理

**核心任务：**
1. 文本分类
2. 命名实体识别
3. 机器翻译
4. 问答系统

Transformer架构彻底改变了NLP领域。

---

## 4. 训练技巧

### 4.1 数据增强

* 图像旋转
* 随机裁剪
* 颜色变换
* 噪声注入

### 4.2 正则化

**常用方法：**
- L1/L2正则化
- Dropout
- Batch Normalization
- Early Stopping

---

## 5. 未来展望

深度学习正朝着以下方向发展：

**技术趋势：**
1. 模型规模持续增大
2. 多模态学习融合
3. 自监督学习突破
4. 高效训练方法

**挑战：**
- 可解释性不足
- 能耗过高
- 数据偏见
- 安全隐私

---

### 参考资源

```bash
# 推荐学习资源
- 论文：Attention Is All You Need
- 课程：Stanford CS231n
- 框架：PyTorch, TensorFlow
```

**总结：** 深度学习是AI的核心驱动力，理解其原理和实践至关重要。"""

    # 创建检测器
    detector = AITextDetector()

    # 测试所有示例
    examples = [
        ("复杂格式AI文本", example1),
        ("纯文本AI输出", example2),
        ("短文本测试", example3),
        ("长文本复杂格式", example4)
    ]

    for name, text in examples:
        print("\n" + "="*60, flush=True)
        print(f"测试: {name}", flush=True)
        print("="*60, flush=True)
        print(f"原始文本（前200字符）:\n{text[:200]}...\n", flush=True)

        result = detector.predict(text, show_analysis=True)

        print("\n" + "="*60, flush=True)
        print("检测结果", flush=True)
        print("="*60, flush=True)
        print(f"预测: {result['prediction']}", flush=True)
        print(f"AI置信度: {result['ai_confidence']:.2%}", flush=True)
        print(f"人类置信度: {result['human_confidence']:.2%}", flush=True)
        print(f"Token数: {result['token_count']}", flush=True)

        # 判断是否需要注意
        if result['format_features']:
            has_format = any([
                result['format_features']['has_bold'],
                result['format_features']['has_heading'],
                result['format_features']['has_list'],
                result['format_features']['has_divider']
            ])

            if has_format:
                print("\n⚠️ 注意: 文本包含markdown格式", flush=True)
                if result['ai_confidence'] > 0.95:
                    print("  → 模型对markdown格式具有良好鲁棒性", flush=True)
                    print("  → 高置信度表明检测基于语义而非格式", flush=True)

        input("\n按Enter继续下一个测试...")


def compare_formatted_vs_plain():
    """对比格式化文本vs纯文本的检测差异"""

    print("\n" + "="*60, flush=True)
    print("格式对比实验", flush=True)
    print("="*60, flush=True)
    print("测试markdown格式是否影响检测准确率\n", flush=True)

    # 同一段话的不同版本
    formatted_text = """## 机器学习简介

**机器学习** 是人工智能的核心技术。

### 主要类型

- **监督学习**: 有标签数据训练
- **无监督学习**: 无标签数据聚类
- **强化学习**: 通过反馈优化策略

---

**应用场景:**
1. 图像识别
2. 语音识别
3. 推荐系统"""

    plain_text = """机器学习简介

机器学习是人工智能的核心技术。

主要类型

监督学习: 有标签数据训练
无监督学习: 无标签数据聚类
强化学习: 通过反馈优化策略

应用场景:
1. 图像识别
2. 语音识别
3. 推荐系统"""

    detector = AITextDetector()

    print("1️⃣ 测试格式化版本", flush=True)
    result1 = detector.predict(formatted_text, show_analysis=True)

    print("\n" + "="*60, flush=True)
    print("检测结果（格式化版本）", flush=True)
    print("="*60, flush=True)
    print(f"预测: {result1['prediction']}", flush=True)
    print(f"AI置信度: {result1['ai_confidence']:.2%}", flush=True)

    input("\n按Enter测试纯文本版本...")

    print("\n2️⃣ 测试纯文本版本", flush=True)
    result2 = detector.predict(plain_text, show_analysis=True)

    print("\n" + "="*60, flush=True)
    print("检测结果（纯文本版本）", flush=True)
    print("="*60, flush=True)
    print(f"预测: {result2['prediction']}", flush=True)
    print(f"AI置信度: {result2['ai_confidence']:.2%}", flush=True)

    # 对比分析
    print("\n" + "="*60, flush=True)
    print("对比分析", flush=True)
    print("="*60, flush=True)

    diff = abs(result1['ai_confidence'] - result2['ai_confidence'])
    print(f"置信度差异: {diff:.2%}", flush=True)

    if diff < 0.05:
        print("✓ 差异很小（<5%），格式对检测影响极小", flush=True)
        print("→ 可以放心测试任何格式的AI文本", flush=True)
    elif diff < 0.10:
        print("⚠️ 有一定差异（5-10%），但仍在可接受范围", flush=True)
    else:
        print("❌ 差异较大（>10%），可能需要预处理", flush=True)


def main():
    parser = argparse.ArgumentParser(description='测试AI生成文本检测器')
    parser.add_argument('--text', type=str, help='要测试的单个文本')
    parser.add_argument('--compare', action='store_true', help='运行格式对比实验')
    parser.add_argument('--examples', action='store_true', help='运行所有示例测试')

    args = parser.parse_args()

    if args.text:
        # 测试单个文本
        detector = AITextDetector()
        result = detector.predict(args.text, show_analysis=True)

        print("\n" + "="*60, flush=True)
        print("检测结果", flush=True)
        print("="*60, flush=True)
        print(f"预测: {result['prediction']}", flush=True)
        print(f"AI置信度: {result['ai_confidence']:.2%}", flush=True)
        print(f"人类置信度: {result['human_confidence']:.2%}", flush=True)

    elif args.compare:
        # 格式对比实验
        compare_formatted_vs_plain()

    elif args.examples:
        # 运行所有示例
        test_with_examples()

    else:
        # 默认：交互式菜单
        print("\n" + "="*60, flush=True)
        print("AI文本检测测试工具", flush=True)
        print("="*60, flush=True)
        print("\n请选择测试模式:", flush=True)
        print("1. 测试示例文本（包含各种markdown格式）", flush=True)
        print("2. 格式对比实验（测试markdown影响）", flush=True)
        print("3. 输入自定义文本", flush=True)

        choice = input("\n选择 (1/2/3): ").strip()

        if choice == '1':
            test_with_examples()
        elif choice == '2':
            compare_formatted_vs_plain()
        elif choice == '3':
            print("\n请输入要测试的文本（输入END结束）:", flush=True)
            lines = []
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)

            text = '\n'.join(lines)
            detector = AITextDetector()
            result = detector.predict(text, show_analysis=True)

            print("\n" + "="*60, flush=True)
            print("检测结果", flush=True)
            print("="*60, flush=True)
            print(f"预测: {result['prediction']}", flush=True)
            print(f"AI置信度: {result['ai_confidence']:.2%}", flush=True)
            print(f"人类置信度: {result['human_confidence']:.2%}", flush=True)
        else:
            print("无效选择", flush=True)


if __name__ == "__main__":
    main()
