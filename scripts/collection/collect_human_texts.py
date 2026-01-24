"""
人类文本数据收集脚本
用于收集高质量的人类撰写中文文本，匹配AI文本的长度和主题分布

目标：收集约9,000条人类文本
- 中文维基百科：~3,000条
- 知乎精华内容：~2,000条
- 新闻语料：~2,000条
- 学术摘要：~1,000条
- 文学作品/博客：~1,000条
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
import requests
import time
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict

class HumanTextCollector:
    """人类文本收集器"""

    def __init__(self, output_dir="datasets/human_texts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 长度筛选标准（匹配AI文本）
        self.min_length = 300
        self.max_length = 3000
        self.target_avg_length = 1777  # 匹配AI文本平均长度

        # 收集计数
        self.collected = {
            'wikipedia': 0,
            'zhihu': 0,
            'news': 0,
            'academic': 0,
            'literature': 0
        }

        print("=" * 60, flush=True)
        print("人类文本收集器初始化完成", flush=True)
        print(f"目标长度范围: {self.min_length}-{self.max_length} 字符", flush=True)
        print(f"目标平均长度: {self.target_avg_length} 字符", flush=True)
        print("=" * 60, flush=True)

    def collect_wikipedia(self, target_count=3000, sleep_s=0.5):
        """
        收集中文维基百科文本

        策略：
        1. 使用Wikipedia API随机获取文章
        2. 提取正文段落
        3. 筛选合适长度的段落
        4. 确保主题多样性
        """
        print(f"\n[维基百科] 开始收集，目标: {target_count} 条", flush=True)

        texts = []
        attempts = 0
        max_attempts = target_count * 3  # 允许3倍尝试次数

        while len(texts) < target_count and attempts < max_attempts:
            attempts += 1

            try:
                # 获取随机维基百科页面
                response = requests.get(
                    'https://zh.wikipedia.org/api/rest_v1/page/random/summary',
                    headers={'User-Agent': 'Mozilla/5.0'},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()

                    # 提取标题和正文
                    title = data.get('title', '')
                    extract = data.get('extract', '')

                    # 清理文本
                    text = self._clean_wikipedia_text(extract)
                    text_length = len(text)

                    # 长度筛选
                    if self.min_length <= text_length <= self.max_length:
                        texts.append({
                            'text': text,
                            'source': 'wikipedia',
                            'title': title,
                            'length': text_length,
                            'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'timestamp': datetime.now().isoformat()
                        })

                        if len(texts) % 100 == 0:
                            print(f"  已收集: {len(texts)}/{target_count} 条 (尝试: {attempts})", flush=True)

                # 避免请求过快
                time.sleep(sleep_s)

            except Exception as e:
                if attempts % 100 == 0:
                    print(f"  错误 (已忽略): {str(e)[:50]}", flush=True)
                time.sleep(1)
                continue

        print(f"[维基百科] 收集完成: {len(texts)} 条", flush=True)
        self.collected['wikipedia'] = len(texts)

        # 保存
        self._save_texts(texts, 'wikipedia')
        return texts

    def collect_baike(self, target_count=2000):
        """
        收集百度百科文本

        作为维基百科的补充来源
        """
        print(f"\n[百度百科] 开始收集，目标: {target_count} 条", flush=True)

        # 主题关键词列表（匹配AI文本的六维属性）
        topics = [
            # 科技类
            "人工智能", "量子计算", "区块链", "云计算", "大数据", "物联网",
            "机器学习", "深度学习", "计算机视觉", "自然语言处理",

            # 社会类
            "教育改革", "医疗保健", "环境保护", "可持续发展", "城市规划",
            "社会保障", "公共卫生", "文化传承", "非物质文化遗产",

            # 经济类
            "数字经济", "共享经济", "绿色经济", "创新创业", "金融科技",
            "电子商务", "供应链管理", "企业管理", "市场营销",

            # 文化艺术类
            "传统艺术", "现代艺术", "音乐理论", "电影艺术", "文学创作",
            "戏剧表演", "舞蹈艺术", "书法艺术", "绘画艺术",

            # 科学类
            "生物科学", "化学工程", "物理学", "天文学", "地理学",
            "心理学", "社会学", "人类学", "考古学",

            # 历史类
            "中国历史", "世界历史", "文化交流", "历史人物", "历史事件"
        ]

        texts = []

        print(f"[百度百科] 将基于 {len(topics)} 个主题关键词收集", flush=True)
        print("  注意：此功能需要进一步实现爬虫逻辑", flush=True)
        print("  建议优先使用维基百科和新闻API", flush=True)

        return texts

    def collect_news_api(self, target_count=2000):
        """
        收集新闻文本

        使用新闻API或RSS源
        """
        print(f"\n[新闻API] 开始收集，目标: {target_count} 条", flush=True)

        # 可用的中文新闻RSS源
        rss_sources = [
            'http://www.people.com.cn/rss/politics.xml',  # 人民网-政治
            'http://www.people.com.cn/rss/tech.xml',       # 人民网-科技
            'http://www.people.com.cn/rss/society.xml',    # 人民网-社会
        ]

        texts = []

        print("[新闻API] RSS源收集功能待实现", flush=True)
        print("  建议使用新闻API或爬虫工具", flush=True)

        return texts

    def _clean_wikipedia_text(self, text: str) -> str:
        """清理维基百科文本"""
        # 删除引用标记 [1], [2] 等
        text = re.sub(r'\[\d+\]', '', text)

        # 删除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 统一空白符
        text = re.sub(r'\s+', ' ', text)

        # 删除特殊字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        return text.strip()

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

        total = sum(self.collected.values())
        print(f"总计收集: {total} 条", flush=True)

        for source, count in self.collected.items():
            if count > 0:
                print(f"  - {source}: {count} 条", flush=True)

        # 加载所有已收集的文本进行分析
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]

        if all_files:
            dfs = []
            for file in all_files:
                try:
                    df = pd.read_csv(f"{self.output_dir}/{file}", encoding='utf-8-sig')
                    dfs.append(df)
                except:
                    pass

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                print(f"\n实际文本总数: {len(combined)} 条", flush=True)
                print(f"平均长度: {combined['length'].mean():.0f} 字符", flush=True)
                print(f"长度范围: {combined['length'].min()} - {combined['length'].max()}", flush=True)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Collect public human texts.")
    parser.add_argument("--wiki-count", type=int, default=3000, help="Wikipedia samples to collect")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests (seconds)")
    parser.add_argument("--min-length", type=int, default=300, help="Minimum length filter")
    parser.add_argument("--max-length", type=int, default=3000, help="Maximum length filter")
    args = parser.parse_args()
    print("人类文本数据收集程序", flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    collector = HumanTextCollector()
    collector.min_length = args.min_length
    collector.max_length = args.max_length

    # 阶段1: 收集维基百科（最可靠的来源）
    print("\n" + "="*60, flush=True)
    print("阶段 1/5: 收集维基百科文本", flush=True)
    print("="*60, flush=True)

    wiki_texts = collector.collect_wikipedia(target_count=args.wiki_count, sleep_s=args.sleep)

    # 生成摘要
    collector.generate_summary()

    print("\n" + "="*60, flush=True)
    print("第一阶段收集完成！", flush=True)
    print("="*60, flush=True)
    print("\n后续阶段:", flush=True)
    print("  2. 知乎精华内容 (~2,000条)", flush=True)
    print("  3. 新闻语料 (~2,000条)", flush=True)
    print("  4. 学术摘要 (~1,000条)", flush=True)
    print("  5. 文学作品/博客 (~1,000条)", flush=True)
    print("\n建议：先验证维基百科数据质量，再继续其他来源", flush=True)

if __name__ == "__main__":
    main()
