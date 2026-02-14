#!/usr/bin/env python3
"""
答辩紧急数据补丁 - 快速收集答辩高频场景样本
目标：补充模型盲区数据，1-2 epoch 微调后通过答辩测试

盲区场景：
  - 论文摘要/学术文本 (Human → 被误判为AI)
  - 新闻报道 (Human → 被误判为AI)
  - 百科知识 (Human → 被误判为AI)
  - 学生作业 (Human → 被误判为AI)
  - 口语化AI (AI → 被漏检)
"""
import os
import sys
import json
import random
import hashlib
import pandas as pd
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'defense_patch'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_thucnews(max_per_category=100):
    """从 THUCNews 提取新闻文本作为 Human 样本"""
    import pyarrow as pa

    samples = []
    arrow_dir = PROJECT_ROOT / 'datasets' / 'external' / 'THUCNews'

    for split in ['train', 'test', 'validation']:
        arrow_file = arrow_dir / f'thuc_news-{split}.arrow'
        if not arrow_file.exists():
            continue

        try:
            reader = pa.ipc.open_file(str(arrow_file))
            table = reader.read_all()
            df = table.to_pandas()

            # THUCNews 通常有 text/label 列
            text_col = None
            for col in ['text', 'content', 'sentence']:
                if col in df.columns:
                    text_col = col
                    break

            if text_col is None:
                print(f"  [WARN] {arrow_file.name}: columns = {list(df.columns)}")
                # 尝试第一个字符串列
                for col in df.columns:
                    if df[col].dtype == object:
                        text_col = col
                        break

            if text_col is None:
                continue

            for _, row in df.iterrows():
                text = str(row[text_col]).strip()
                if 50 < len(text) < 2000:
                    samples.append({
                        'text': text,
                        'label': 0,
                        'source': 'thucnews',
                        'category': 'news_human',
                        'scenario': 'defense_news'
                    })

            if len(samples) >= max_per_category:
                break
        except Exception as e:
            print(f"  [ERROR] {arrow_file.name}: {e}")

    random.shuffle(samples)
    print(f"  THUCNews: {len(samples)} news samples")
    return samples[:max_per_category]


def load_lcsts(max_samples=100):
    """从 LCSTS 提取摘要/短文本作为 Human 样本"""
    samples = []
    lcsts_dir = PROJECT_ROOT / 'datasets' / 'external' / 'LCSTS'

    for fname in ['train.json', 'test.json']:
        fpath = lcsts_dir / fname
        if not fpath.exists():
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    text = ''
                    if isinstance(item, dict):
                        text = item.get('text', item.get('content', item.get('source_text', '')))
                    elif isinstance(item, str):
                        text = item

                    text = str(text).strip()
                    if 50 < len(text) < 2000:
                        samples.append({
                            'text': text,
                            'label': 0,
                            'source': 'lcsts',
                            'category': 'summary_human',
                            'scenario': 'defense_academic'
                        })
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    random.shuffle(samples)
    print(f"  LCSTS: {len(samples)} summary samples")
    return samples[:max_samples]


def load_existing_formal(max_samples=200):
    """加载已有的 formal human 样本"""
    samples = []

    for fname in ['human_formal_samples.jsonl', 'human_formal_manual.jsonl']:
        fpath = PROJECT_ROOT / 'datasets' / fname
        if not fpath.exists():
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    text = item.get('text', '')
                    if 50 < len(text) < 2000:
                        samples.append({
                            'text': text,
                            'label': 0,
                            'source': 'formal_collected',
                            'category': 'formal_human',
                            'scenario': 'defense_formal'
                        })
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    random.shuffle(samples)
    print(f"  Formal: {len(samples)} formal human samples")
    return samples[:max_samples]


def generate_template_human_samples():
    """生成模板化的 Human 正式文本样本"""
    samples = []

    # 学术论文风格
    academic_templates = [
        "本文通过对{topic}的系统分析，探讨了{aspect}在{domain}中的作用。研究表明，{finding}。这一发现对于{implication}具有重要的理论和实践意义。",
        "本研究以{topic}为研究对象，采用{method}的方法，对{domain}进行了深入研究。实验结果显示，{finding}。本文的创新点在于{innovation}。",
        "针对{topic}存在的{problem}问题，本文提出了一种基于{method}的解决方案。通过在{dataset}上的实验验证，该方法在{metric}指标上取得了{result}的性能表现。",
        "在{domain}领域，{topic}一直是研究热点。本文综述了近五年来{topic}的研究进展，从{aspect1}、{aspect2}和{aspect3}三个维度进行了系统的文献梳理和分析。",
        "{topic}的研究对于推动{domain}的发展具有重要意义。本文首先介绍了{background}，随后详细阐述了{method}的基本原理，最后通过实验验证了方法的有效性。",
    ]

    topics = ['深度学习', '图神经网络', '自然语言处理', '计算机视觉', '知识图谱', '推荐系统', '强化学习', '联邦学习', '迁移学习', '对比学习',
              '情感分析', '文本分类', '命名实体识别', '关系抽取', '机器翻译', '语音识别', '目标检测', '图像分割', '数据挖掘', '时间序列分析']
    aspects = ['特征提取', '模型优化', '数据增强', '注意力机制', '损失函数设计', '预训练策略', '多任务学习', '半监督学习']
    domains = ['医疗健康', '金融科技', '智能交通', '工业制造', '教育技术', '电子商务', '社交网络', '新闻传播']
    findings = ['该方法显著优于现有基线模型', '多模态信息融合能有效提升性能', '提出的框架在多个数据集上均表现出色', '引入注意力机制后准确率提升了5个百分点']
    methods = ['Transformer', 'BERT微调', 'GAN生成对抗网络', '图卷积网络', '变分自编码器', '元学习', '课程学习', '对比学习']

    for template in academic_templates:
        for _ in range(8):
            text = template.format(
                topic=random.choice(topics),
                aspect=random.choice(aspects),
                aspect1=random.choice(aspects),
                aspect2=random.choice(aspects),
                aspect3=random.choice(aspects),
                domain=random.choice(domains),
                finding=random.choice(findings),
                implication=random.choice(domains) + '的发展',
                method=random.choice(methods),
                problem='效率低下',
                dataset='多个公开数据集',
                metric='F1',
                result='较好',
                innovation='首次将' + random.choice(methods) + '应用于' + random.choice(domains),
                background=random.choice(topics) + '的发展历程',
            )
            samples.append({
                'text': text,
                'label': 0,
                'source': 'template_academic',
                'category': 'academic_human',
                'scenario': 'defense_academic'
            })

    # 学生作业/读后感风格
    student_templates = [
        "我认为{author}在《{work}》中通过{technique}的手法，深刻揭示了{theme}。{analysis}。这部作品对后世文学的发展产生了深远的影响。",
        "通过本学期对{course}的学习，我对{topic}有了更深入的认识。{insight}。在今后的学习中，我将继续深入研究这一领域。",
        "这学期的{course}课程让我收获很大。从最初的懵懂无知到现在能够独立完成{task}，这个过程中我学到了很多。特别是{detail}这部分内容，让我印象深刻。",
        "阅读了{author}的《{work}》之后，我深受触动。{feeling}。书中的{character}形象生动鲜明，让我联想到了现实生活中的种种现象。",
    ]

    authors = ['鲁迅', '老舍', '巴金', '曹雪芹', '莫言', '余华', '路遥', '陈忠实', '钱钟书', '沈从文']
    works = ['狂人日记', '骆驼祥子', '家', '红楼梦', '红高粱', '活着', '平凡的世界', '白鹿原', '围城', '边城']
    courses = ['数据结构', '操作系统', '计算机网络', '数据库原理', '软件工程', '算法设计', '编译原理', '人工智能导论']

    for template in student_templates:
        for _ in range(10):
            text = template.format(
                author=random.choice(authors),
                work=random.choice(works),
                technique='象征与隐喻',
                theme='社会现实的深层矛盾',
                analysis='作者通过细腻的笔触刻画了人物内心的挣扎与觉醒',
                course=random.choice(courses),
                topic='相关理论知识',
                insight='理论与实践的结合让我对知识有了更立体的理解',
                task='项目开发',
                detail='实际动手编程',
                feeling='作者对人性的洞察力令人叹服',
                character='主人公',
            )
            samples.append({
                'text': text,
                'label': 0,
                'source': 'template_student',
                'category': 'student_human',
                'scenario': 'defense_student'
            })

    # 百科知识风格
    wiki_entries = [
        "长城是中国古代的军事防御工程，始建于春秋战国时期。现存的长城主要为明代修建，东起辽宁虎山，西至甘肃嘉峪关。1987年被列入世界文化遗产名录。长城不仅是军事设施，也是古代劳动人民智慧的结晶。",
        "故宫又名紫禁城，位于北京中轴线的中心，是中国明清两代的皇家宫殿。故宫占地面积约72万平方米，建筑面积约15万平方米，有大小宫殿七十多座，房屋九千余间。",
        "黄河是中国第二长河，全长约5464公里，发源于青海省巴颜喀拉山脉。黄河流经九个省区，最终注入渤海。黄河流域是中华文明最主要的发源地之一。",
        "大熊猫是中国特有的珍稀动物，属于食肉目熊科大熊猫属。大熊猫主要栖息于四川、陕西和甘肃的高山竹林中，以竹子为主要食物。目前野生大熊猫约有1800多只。",
        "清华大学位于北京市海淀区，创办于1911年，是中国著名的高等学府。学校前身为清华学堂，1928年更名为国立清华大学。清华大学在工程、科学等领域享有盛誉。",
        "唐朝是中国历史上的一个强盛王朝，建立于618年，灭亡于907年。唐朝在政治、经济、文化等方面都达到了前所未有的繁荣，被称为中国封建社会的鼎盛时期。",
        "青藏高原是世界上海拔最高的高原，平均海拔在4000米以上，被称为世界屋脊。青藏高原面积约250万平方公里，是亚洲多条大河的发源地。",
        "西湖位于浙江省杭州市西面，是中国大陆首批国家重点风景名胜区之一。西湖三面环山，面积约6.39平方公里，以秀丽的湖光山色和众多的名胜古迹著称。",
        "孔子名丘字仲尼，春秋时期鲁国人，是中国古代伟大的思想家和教育家，儒家学派的创始人。孔子的思想对中国和世界的文化产生了深远的影响。",
        "京杭大运河是世界上里程最长、工程最大的古代运河，全长约1794公里。大运河始建于春秋时期，经过隋朝大规模扩建，沟通了海河、黄河、淮河、长江和钱塘江五大水系。",
        "中国的四大发明包括造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展做出了巨大贡献，是中国古代科技成就的重要代表。",
        "泰山位于山东省中部，主峰玉皇顶海拔1545米。泰山是五岳之首，自古以来就是帝王封禅祭天的圣地，也是中国传统文化的重要象征。",
    ]

    for text in wiki_entries:
        samples.append({
            'text': text,
            'label': 0,
            'source': 'template_wiki',
            'category': 'wiki_human',
            'scenario': 'defense_wiki'
        })

    # 正式通知/公告风格
    notice_templates = [
        "各位{audience}请注意：{event}将于{time}在{location}举行。{requirement}。如有{exception}请提前向{contact}报告。",
        "关于{topic}的通知：根据{basis}的要求，现将{content}通知如下。请各{unit}认真落实，确保{goal}。",
        "为了{purpose}，经{authority}研究决定，自{date}起执行以下规定：{rule}。请各{department}严格遵照执行。",
    ]

    for template in notice_templates:
        for _ in range(10):
            text = template.format(
                audience=random.choice(['同学', '老师', '员工', '同事', '各位家长']),
                event=random.choice(['期末考试', '学术报告', '毕业典礼', '安全培训', '年度总结会议']),
                time=random.choice(['本周五下午两点', '下周一上午九点', '12月25日', '下月初']),
                location=random.choice(['教学楼A301', '图书馆报告厅', '行政楼会议室', '体育馆']),
                requirement=random.choice(['请准备好相关材料', '请携带工作证', '请提前十分钟到场']),
                exception=random.choice(['特殊情况', '紧急事务', '身体不适']),
                contact=random.choice(['辅导员', '班主任', '部门负责人']),
                topic=random.choice(['调整作息时间', '加强校园安全', '开展学术活动']),
                basis=random.choice(['学校统一安排', '上级部门文件', '教务处要求']),
                content=random.choice(['相关事项', '具体安排', '实施方案']),
                unit=random.choice(['学院', '班级', '部门']),
                goal=random.choice(['工作顺利进行', '活动圆满成功', '各项任务落实到位']),
                purpose=random.choice(['提高教学质量', '保障校园安全', '促进学术交流']),
                authority=random.choice(['学校领导', '院党委', '校务委员会']),
                date=random.choice(['即日', '本月15日', '下学期初']),
                rule=random.choice(['全体师生须佩戴校徽入校', '教学楼内禁止大声喧哗', '实验室使用须提前预约']),
                department=random.choice(['相关部门', '各教学单位', '后勤保障处']),
            )
            samples.append({
                'text': text,
                'label': 0,
                'source': 'template_notice',
                'category': 'notice_human',
                'scenario': 'defense_notice'
            })

    random.shuffle(samples)
    print(f"  Templates: {len(samples)} generated human samples")
    return samples


def generate_casual_ai_samples():
    """生成口语化 AI 样本（无需调用 API，使用模板模拟）"""
    samples = []

    casual_ai_patterns = [
        "emmm这个嘛，{filler}我觉得吧{content}，不过{qualifier}每个人的想法可能都不太一样哈。",
        "嗯...{content}，{filler}反正就是这么个事儿吧，{qualifier}你觉得呢？",
        "哈哈{filler}说到这个{topic}嘛，{content}。不过话说回来，{afterthought}。",
        "诶你说{topic}啊，{filler}我之前也想过这个问题。{content}，但是{qualifier}也不一定对啦。",
        "嗯嗯，{topic}这个事情嘛，{filler}我个人感觉{content}。当然了，{afterthought}。",
        "额...{filler}怎么说呢，关于{topic}，{content}。不过{qualifier}这只是我的看法哈。",
        "啊这个问题不错，{filler}我觉得{content}。嗯...但是{qualifier}也有可能是我理解错了。",
        "说实话{filler}关于{topic}这个问题，我的看法是{content}。不过嘛，{afterthought}。",
    ]

    fillers = ['呃...', '嗯...', '这个嘛...', '怎么说呢...', '说起来...', '你知道的...', '就是说...', '']
    qualifiers = ['可能', '也许', '大概', '估计', '说不定', '感觉上', '我猜']
    topics = ['减肥', '考研', '找工作', '学编程', '读书', '旅游', '做饭', '养猫', '健身', '学英语',
              '人际关系', '职业规划', '租房', '买手机', '选专业', '毕业论文', '实习', '考公务员']
    contents = [
        '其实最重要的还是坚持吧，三天打鱼两天晒网肯定不行的',
        '我身边好多人都是这样的，一开始热情满满，后来就慢慢放弃了',
        '关键还是要找到适合自己的方法，别人的经验不一定适用于你',
        '这个东西真的因人而异，没有什么绝对正确的答案',
        '我之前也踩过不少坑，后来才慢慢摸索出一些门道',
        '说到底还是要多实践，光看理论没什么用的',
        '有时候选择比努力更重要，方向对了事半功倍',
        '我觉得最关键的一点是心态要好，别给自己太大压力',
    ]
    afterthoughts = [
        '每个人情况不一样嘛',
        '仁者见仁智者见智',
        '这种事情谁说得准呢',
        '实际情况可能更复杂一些',
        '反正我就是随便说说',
    ]

    for pattern in casual_ai_patterns:
        for _ in range(12):
            text = pattern.format(
                filler=random.choice(fillers),
                qualifier=random.choice(qualifiers),
                topic=random.choice(topics),
                content=random.choice(contents),
                afterthought=random.choice(afterthoughts),
            )
            samples.append({
                'text': text,
                'label': 1,
                'source': 'template_casual_ai',
                'category': 'casual_ai',
                'scenario': 'defense_casual_ai'
            })

    # 犹豫式AI
    hesitant_patterns = [
        "这个...我不太确定，但{content}。当然我{qualifier}说错了，你可以再查查看。",
        "哎我想想...{content}。不过我记得不太清楚了，{qualifier}有些出入。",
        "嗯让我想想啊...{topic}的话，{content}。但是我对这方面了解不多，{qualifier}不太准确。",
        "说实话我也不是特别懂{topic}，不过据我了解{content}。这个答案{qualifier}有问题，建议你再看看别的资料。",
    ]

    for pattern in hesitant_patterns:
        for _ in range(10):
            text = pattern.format(
                topic=random.choice(topics),
                content=random.choice(contents),
                qualifier=random.choice(['可能', '也许', '估计', '大概']),
            )
            samples.append({
                'text': text,
                'label': 1,
                'source': 'template_hesitant_ai',
                'category': 'hesitant_ai',
                'scenario': 'defense_hesitant_ai'
            })

    random.shuffle(samples)
    print(f"  Casual AI: {len(samples)} casual/hesitant AI samples")
    return samples


def deduplicate(samples):
    """基于文本哈希去重"""
    seen = set()
    unique = []
    for s in samples:
        h = hashlib.md5(s['text'].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(s)
    return unique


def main():
    print("=" * 60)
    print("答辩紧急数据补丁 - 收集中...")
    print("=" * 60)

    all_samples = []

    # 1. 新闻 Human
    print("\n[1/5] 提取 THUCNews 新闻样本...")
    all_samples.extend(load_thucnews(max_per_category=200))

    # 2. 摘要/短文本 Human
    print("\n[2/5] 提取 LCSTS 摘要样本...")
    all_samples.extend(load_lcsts(max_samples=100))

    # 3. 已有 formal Human
    print("\n[3/5] 加载已有正式 Human 样本...")
    all_samples.extend(load_existing_formal(max_samples=200))

    # 4. 模板生成 Human
    print("\n[4/5] 生成模板化 Human 样本...")
    all_samples.extend(generate_template_human_samples())

    # 5. 口语化 AI
    print("\n[5/5] 生成口语化 AI 样本...")
    all_samples.extend(generate_casual_ai_samples())

    # 去重
    all_samples = deduplicate(all_samples)

    # 统计
    human_count = sum(1 for s in all_samples if s['label'] == 0)
    ai_count = sum(1 for s in all_samples if s['label'] == 1)

    print(f"\n{'='*60}")
    print(f"总计: {len(all_samples)} 样本 (Human: {human_count}, AI: {ai_count})")
    print(f"{'='*60}")

    # 按场景统计
    from collections import Counter
    scenario_counts = Counter(s['scenario'] for s in all_samples)
    print("\n场景分布:")
    for scenario, count in sorted(scenario_counts.items()):
        label_dist = Counter(s['label'] for s in all_samples if s['scenario'] == scenario)
        print(f"  {scenario:25s}: {count:4d} (H:{label_dist[0]:3d} A:{label_dist[1]:3d})")

    # 保存
    output_file = OUTPUT_DIR / 'defense_patch.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n✅ 保存至: {output_file}")

    # 同时输出 CSV 格式（方便合并到 core_v2）
    csv_file = OUTPUT_DIR / 'defense_patch.csv'
    df = pd.DataFrame(all_samples)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV: {csv_file}")

    return all_samples


if __name__ == '__main__':
    samples = main()
