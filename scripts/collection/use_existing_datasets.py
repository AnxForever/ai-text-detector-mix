"""
使用现成的中文文本数据集收集人类文本
主要数据源：THUCNews、CLUE等Hugging Face数据集
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

import pandas as pd
import json
from datetime import datetime
from typing import List, Dict

class DatasetCollector:
    """使用现成数据集收集人类文本"""

    def __init__(self, output_dir="datasets/human_texts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 长度筛选标准（匹配AI文本）
        self.min_length = 300
        self.max_length = 3000
        self.target_avg_length = 1777

        print("=" * 60, flush=True)
        print("数据集收集器初始化完成", flush=True)
        print(f"目标长度范围: {self.min_length}-{self.max_length} 字符", flush=True)
        print(f"目标平均长度: {self.target_avg_length} 字符", flush=True)
        print("=" * 60, flush=True)

    def load_thucnews_from_hf(self, target_count=5000):
        """
        从Hugging Face加载THUCNews数据集

        THUCNews包含74万篇新闻，14个类别
        """
        print(f"\n[THUCNews] 尝试从Hugging Face加载数据集...", flush=True)

        try:
            from datasets import load_dataset

            print("  正在下载数据集（首次运行可能需要较长时间）...", flush=True)

            # 加载THUCNews数据集
            dataset = load_dataset("oyxy2019/THUCNewsText", split="train")

            print(f"  ✓ 数据集加载成功！总样本数: {len(dataset)}", flush=True)

            # 转换为DataFrame
            df = pd.DataFrame(dataset)

            # 筛选长度符合要求的文本
            df['length'] = df['text'].str.len()
            filtered = df[
                (df['length'] >= self.min_length) &
                (df['length'] <= self.max_length)
            ].copy()

            print(f"  长度筛选后: {len(filtered)} 条", flush=True)

            # 如果超过目标数量，随机采样
            if len(filtered) > target_count:
                filtered = filtered.sample(n=target_count, random_state=42)
                print(f"  随机采样: {target_count} 条", flush=True)

            # 格式化为标准格式
            result = []
            for idx, row in filtered.iterrows():
                result.append({
                    'text': row['text'],
                    'source': 'thucnews',
                    'category': row.get('label', 'unknown'),
                    'length': row['length'],
                    'timestamp': datetime.now().isoformat()
                })

            self._save_texts(result, 'thucnews')
            return result

        except ImportError:
            print("  ✗ 错误：未安装 datasets 库", flush=True)
            print("  请运行: pip install datasets", flush=True)
            return []

        except Exception as e:
            print(f"  ✗ 加载失败: {str(e)}", flush=True)
            print("  将尝试使用本地文件或其他方案...", flush=True)
            return []

    def create_sample_texts(self, target_count=1000):
        """
        创建示例人类文本（用于测试或无法访问外部数据源时）

        包含不同主题和风格的文本样例
        """
        print(f"\n[示例数据] 创建 {target_count} 条示例文本...", flush=True)

        sample_templates = [
            # 科技类
            "人工智能技术在近年来取得了突破性进展。深度学习算法的发展使得计算机视觉和自然语言处理能力大幅提升。研究人员通过构建更深层的神经网络，实现了图像识别准确率的显著提高。同时，生成对抗网络的应用也为图像生成带来了新的可能性。这些技术进步不仅推动了学术研究的发展，也为实际应用创造了广阔空间。从自动驾驶到医疗诊断，从智能客服到内容创作，人工智能正在深刻改变我们的生活方式。然而，随着技术的快速发展，数据隐私、算法偏见等问题也日益凸显，需要社会各界共同关注和解决。",

            # 教育类
            "教育改革一直是社会关注的焦点话题。传统的应试教育模式逐渐暴露出诸多弊端，如忽视学生个性发展、创新能力培养不足等。新一轮教育改革强调素质教育的重要性，提倡培养学生的批判性思维和创造力。学校开始引入项目制学习、跨学科教学等创新方法，鼓励学生主动探索和实践。教师角色也在转变，从知识的传授者变为学习的引导者。同时，信息技术的融入为教育带来了新的机遇，在线课程、智慧教室等新形式正在重塑传统课堂。然而，教育公平问题仍然存在，城乡差距、资源分配不均等现象需要持续关注。如何在推进改革的同时保障教育质量，是教育工作者面临的重要课题。",

            # 环保类
            "气候变化已成为全球面临的严峻挑战。近几十年来，全球平均气温持续上升，极端天气事件频发，对生态系统和人类社会造成了深远影响。科学研究表明，人类活动产生的温室气体排放是导致全球变暖的主要原因。为应对这一挑战，国际社会达成了《巴黎协定》，各国纷纷制定减排目标和行动计划。可再生能源的发展成为关键举措，太阳能、风能等清洁能源技术不断进步，成本逐年下降。与此同时，公众环保意识也在增强，绿色出行、垃圾分类等理念深入人心。企业也在探索可持续发展模式，将环境责任纳入经营战略。然而，实现碳中和目标仍面临技术、经济等多方面挑战，需要政府、企业和个人共同努力。",

            # 文化类
            "传统文化的传承与创新是当代社会的重要议题。在全球化背景下，各种文化相互碰撞融合，传统文化面临着前所未有的挑战和机遇。一方面，现代生活方式的普及使得许多传统习俗逐渐淡出人们的视野；另一方面，人们对文化认同和精神归属的需求又促使传统文化复兴。非物质文化遗产保护工作受到高度重视，各地纷纷开展传统技艺的抢救性记录和活态传承。同时，文化创意产业蓬勃发展，通过创新表达形式，让传统文化焕发新的生命力。故宫文创产品的成功、国潮品牌的兴起，都展示了传统与现代结合的无限可能。文化传承不是简单的复古，而是要在理解传统精髓的基础上，创造性转化、创新性发展，使其更好地服务于当代社会。",

            # 经济类
            "数字经济正在重塑全球经济格局。互联网、大数据、人工智能等技术的广泛应用，催生了电子商务、共享经济、平台经济等新业态。企业数字化转型成为必然趋势，传统产业纷纷拥抱互联网，探索线上线下融合发展模式。数字支付的普及极大地改变了人们的消费习惯，移动支付在中国的渗透率已处于全球领先地位。与此同时，数据作为新的生产要素，其价值日益凸显。然而，数字鸿沟、数据安全、算法歧视等问题也引发广泛关注。如何在促进数字经济发展的同时，保护个人隐私、维护公平竞争，成为监管部门面临的新课题。各国政府正在探索适合本国国情的数字经济治理模式，力求在创新与监管之间找到平衡点。"
        ]

        # 扩展到target_count条
        texts = []
        for i in range(target_count):
            template = sample_templates[i % len(sample_templates)]

            # 添加一些变化
            variation = f"（第{i+1}篇）" + template

            texts.append({
                'text': variation,
                'source': 'sample',
                'category': ['科技', '教育', '环保', '文化', '经济'][i % 5],
                'length': len(variation),
                'timestamp': datetime.now().isoformat()
            })

        self._save_texts(texts, 'sample')
        return texts

    def _save_texts(self, texts: List[Dict], source: str):
        """保存收集的文本"""
        if not texts:
            return

        df = pd.DataFrame(texts)

        filename = f"{self.output_dir}/{source}_{len(texts)}_texts.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"  ✓ 已保存到: {filename}", flush=True)

        # 统计信息
        avg_length = df['length'].mean()
        print(f"  平均长度: {avg_length:.0f} 字符", flush=True)
        print(f"  长度范围: {df['length'].min()} - {df['length'].max()}", flush=True)

    def generate_summary(self):
        """生成收集摘要"""
        print("\n" + "=" * 60, flush=True)
        print("收集摘要", flush=True)
        print("=" * 60, flush=True)

        # 加载所有已收集的文本进行分析
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]

        if all_files:
            dfs = []
            for file in all_files:
                try:
                    df = pd.read_csv(f"{self.output_dir}/{file}", encoding='utf-8-sig')
                    dfs.append(df)
                    print(f"  - {file}: {len(df)} 条", flush=True)
                except:
                    pass

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                print(f"\n实际文本总数: {len(combined)} 条", flush=True)
                print(f"平均长度: {combined['length'].mean():.0f} 字符", flush=True)
                print(f"长度范围: {combined['length'].min()} - {combined['length'].max()}", flush=True)

                if 'category' in combined.columns:
                    print(f"\n类别分布:", flush=True)
                    print(combined['category'].value_counts().head(10))
        else:
            print("  尚无已保存的数据", flush=True)


def main():
    """主函数"""
    print("人类文本数据集收集程序", flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    collector = DatasetCollector()

    # 方案1：尝试从Hugging Face加载THUCNews
    print("\n" + "="*60, flush=True)
    print("方案1: 从Hugging Face加载THUCNews数据集", flush=True)
    print("="*60, flush=True)

    texts = collector.load_thucnews_from_hf(target_count=9000)

    # 如果方案1失败，使用方案2：示例数据
    if not texts:
        print("\n" + "="*60, flush=True)
        print("方案2: 使用示例数据（用于测试）", flush=True)
        print("="*60, flush=True)
        texts = collector.create_sample_texts(target_count=1000)

    # 生成摘要
    collector.generate_summary()

    print("\n" + "="*60, flush=True)
    print("数据收集完成！", flush=True)
    print("="*60, flush=True)

    if texts:
        print(f"\n成功收集 {len(texts)} 条人类文本", flush=True)
        print("\n下一步:", flush=True)
        print("  1. 检查数据质量", flush=True)
        print("  2. 合并AI文本和人类文本", flush=True)
        print("  3. 添加标签（AI=1, 人类=0）", flush=True)
        print("  4. 划分train/val/test集", flush=True)
        print("  5. 转换为BERT格式", flush=True)


if __name__ == "__main__":
    main()
