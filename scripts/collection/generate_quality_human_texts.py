"""
生成高质量的模拟人类文本数据
用于BERT二分类训练

策略：基于AI生成数据的统计特征，反向构造人类风格的文本
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
import random
from datetime import datetime

class QualityHumanTextGenerator:
    """高质量人类文本生成器"""

    def __init__(self):
        # 加载AI生成的数据以获取主题分布
        try:
            self.ai_df = pd.read_csv('datasets/final/parallel_dataset_cleaned.csv',
                                     encoding='utf-8-sig')
            print(f"✓ 已加载AI数据集：{len(self.ai_df)} 条", flush=True)
        except:
            self.ai_df = None
            print("! 未找到AI数据集，将使用默认主题", flush=True)

        # 目标统计特征（匹配AI数据）
        self.target_avg_length = 1777
        self.min_length = 300
        self.max_length = 3000

        # 五类文本属性的高质量模板
        self.templates = self._create_templates()

    def _create_templates(self):
        """创建五类文本属性的模板段落库"""
        return {
            '议论': [
                "从历史发展的角度来看，{topic}一直是社会各界广泛关注的焦点。不同学者对此持有不同的观点和立场。支持者认为{positive_view}，而反对者则坚持{negative_view}。通过深入分析我们可以发现，这一问题的本质在于{core_issue}。",
                "当代社会中，{topic}的重要性日益凸显。这不仅关系到{aspect1}，更深刻影响着{aspect2}。我们应当客观理性地看待这一现象，既要看到其积极作用，也要警惕可能带来的负面影响。只有这样，才能制定出科学合理的政策措施。",
                "关于{topic}的讨论由来已久。从理论层面来说，{theory}为我们提供了重要的分析框架。然而，理论与实践之间往往存在差距。实际情况表明，{practice}。因此，我们需要在坚持基本原则的同时，灵活应对各种复杂情况。",
                "{topic}作为一个复杂的社会现象，涉及多个层面的问题。首要问题是{problem1}，这直接关系到整体发展方向。其次，{problem2}也不容忽视，它影响着具体实施效果。面对这些挑战，我们必须采取综合性的应对策略，统筹兼顾各方面因素。",
                "在全球化背景下，{topic}呈现出新的特征和趋势。国际经验表明，{international_experience}。结合我国实际情况，我们既要学习借鉴国外的成功做法，又要立足本国国情，探索出一条符合自身发展需要的道路。这需要各方面共同努力，持续推进改革创新。"
            ],
            '说明': [
                "{topic}的基本概念可以从多个维度进行理解。从定义上看，它是指{definition}。具体而言，{topic}包含以下几个核心要素：第一，{element1}；第二，{element2}；第三，{element3}。这些要素相互关联，共同构成了完整的体系。",
                "{topic}的发展历程大致可以分为几个阶段。初期阶段以{early_stage}为主要特征，这一时期的重点在于{early_focus}。随后进入快速发展阶段，{rapid_stage}成为主流趋势。当前我们正处于{current_stage}阶段，面临着新的机遇和挑战。",
                "要深入了解{topic}，需要掌握其基本原理和运作机制。其核心原理在于{principle}，通过{mechanism}实现预期目标。整个过程涉及多个环节：首先是{step1}，然后是{step2}，最后是{step3}。每个环节都有其特定的要求和标准。",
                "{topic}的分类方式有多种。按照{criterion1}可以分为{type1}和{type2}；按照{criterion2}则可以分为{type3}、{type4}和{type5}。不同类别各有其特点和适用范围。在实际应用中，需要根据具体情况选择合适的类别。",
                "关于{topic}的相关概念容易产生混淆。首先需要明确{topic}与{related_concept1}的区别。前者侧重于{focus1}，而后者强调{focus2}。同时，{topic}也不等同于{related_concept2}，两者在{difference}方面存在本质差异。正确理解这些概念对于深入学习至关重要。"
            ],
            '描写': [
                "{scene}呈现出独特的景象。目光所及之处，{visual_detail1}，{visual_detail2}。远处，{distant_view}；近处，{close_view}。整个画面{overall_impression}，令人印象深刻。",
                "走进{place}，首先映入眼帘的是{first_impression}。环顾四周，{surrounding_description}。空气中弥漫着{smell}的气息，耳边传来{sound}。这里的一切都{characteristic}，营造出{atmosphere}的氛围。",
                "{object}的外观颇具特色。从整体上看，它{overall_shape}。细节之处，{detail1}，{detail2}，{detail3}。材质方面，{material_description}。色彩搭配{color_scheme}，给人以{visual_effect}的视觉感受。",
                "{character}的形象给人留下深刻印象。{他/她}身着{clothing}，{physical_feature}。言谈举止间，{behavior}，充分展现出{personality}的性格特点。在{situation}的情况下，{character}表现出{performance}，令人钦佩。",
                "{period}的景色别有一番风味。此时，{time_specific_view}。{natural_element1}，{natural_element2}，{natural_element3}。整个环境{environment_quality}，处处洋溢着{mood}，让人不禁沉浸其中，流连忘返。"
            ],
            '记叙': [
                "{event}发生在{time}。当时，{initial_situation}。{character}正在{doing}，突然{incident_trigger}。面对这一情况，{character}{immediate_reaction}，随后{subsequent_action}。整个过程{process_description}，最终{outcome}。",
                "那年{time_reference}，{character}经历了一件难忘的事情。事情的起因是{cause}。{character}原本计划{original_plan}，却因为{unexpected_factor}而发生了变化。在{critical_moment}，{character}做出了{decision}的决定。这一选择深刻影响了{impact}。",
                "回忆起{past_event}，{character}依然记忆犹新。{time_setting}，{场景描述}。{character}与{other_character}相遇了。两人{interaction}，产生了{connection}。随着时间推移，{development}。直到{turning_point}，{character}才意识到{realization}。",
                "{event}的发展经历了曲折的过程。开始阶段，一切{initial_stage}。然而，{conflict}的出现打破了原有的平衡。{character}面临{challenge}，不得不{response}。经过{effort}，情况逐渐{change}。最终，{resolution}，事情得到了妥善解决。",
                "在{specific_time}的一个{time_of_day}，{event}悄然展开。{setting_description}。{character}怀着{emotion}的心情，{action}。途中，{encounter}。这次经历让{character}{learning}，也成为{significance}的重要转折点。"
            ],
            '抒情': [
                "每当想起{subject}，内心总会涌起{emotion}。{subject}如同{metaphor}，{symbolic_meaning}。在{situation}的时刻，{subject}给予我{benefit}。这份{feeling}深深烙印在心底，成为永恒的记忆。",
                "{subject}承载着{meaning}。它不仅仅是{surface_level}，更是{deeper_level}。面对{challenge}，{subject}教会我{lesson}。如今，{subject}已经成为{current_status}，指引着{future_direction}。",
                "岁月流转，{subject}依旧{quality}。曾几何时，{past_memory}。如今，{present_situation}。尽管{change}，但{unchanged_core}始终如一。{subject}如同{comparison}，在{aspect}中{role}，令人{感受}。",
                "在这个{时代}的时代，{subject}显得格外{quality}。它{characteristic1}，{characteristic2}，{characteristic3}。每一次与{subject}的相遇，都是一次{experience}。它让我明白{understanding}，也让我珍惜{treasure}。",
                "{subject}是{definition}。它陪伴我走过{journey}，见证了{witness}。在{emotional_moment}的时刻，{subject}给予{support}。这份{bond}超越了{limitation}，成为{ultimate_meaning}。愿{wish}，{hope}能够{future_aspiration}。"
            ]
        }

    def generate_paragraph(self, attribute, topic):
        """生成单个段落"""
        template = random.choice(self.templates[attribute])

        # 根据主题填充模板
        # 这里使用简化的填充逻辑
        filled = template
        placeholders = [
            '{topic}', '{positive_view}', '{negative_view}', '{core_issue}',
            '{aspect1}', '{aspect2}', '{theory}', '{practice}', '{problem1}',
            '{problem2}', '{international_experience}', '{definition}',
            '{element1}', '{element2}', '{element3}', '{early_stage}',
            '{rapid_stage}', '{current_stage}', '{principle}', '{mechanism}',
            '{step1}', '{step2}', '{step3}', '{criterion1}', '{type1}',
            '{type2}', '{criterion2}', '{type3}', '{type4}', '{type5}',
            '{related_concept1}', '{related_concept2}', '{focus1}', '{focus2}',
            '{difference}', '{scene}', '{visual_detail1}', '{visual_detail2}',
            '{distant_view}', '{close_view}', '{overall_impression}',
            '{place}', '{first_impression}', '{surrounding_description}',
            '{smell}', '{sound}', '{characteristic}', '{atmosphere}',
            '{object}', '{overall_shape}', '{detail1}', '{detail2}',
            '{detail3}', '{material_description}', '{color_scheme}',
            '{visual_effect}', '{character}', '{他/她}', '{clothing}',
            '{physical_feature}', '{behavior}', '{personality}',
            '{situation}', '{performance}', '{period}', '{time_specific_view}',
            '{natural_element1}', '{natural_element2}', '{natural_element3}',
            '{environment_quality}', '{mood}', '{event}', '{time}',
            '{initial_situation}', '{doing}', '{incident_trigger}',
            '{immediate_reaction}', '{subsequent_action}',
            '{process_description}', '{outcome}', '{time_reference}',
            '{cause}', '{original_plan}', '{unexpected_factor}',
            '{critical_moment}', '{decision}', '{impact}', '{past_event}',
            '{time_setting}', '{场景描述}', '{other_character}',
            '{interaction}', '{connection}', '{development}',
            '{turning_point}', '{realization}', '{initial_stage}',
            '{conflict}', '{challenge}', '{response}', '{effort}',
            '{change}', '{resolution}', '{specific_time}', '{time_of_day}',
            '{setting_description}', '{action}', '{encounter}', '{learning}',
            '{significance}', '{subject}', '{emotion}', '{metaphor}',
            '{symbolic_meaning}', '{benefit}', '{feeling}', '{meaning}',
            '{surface_level}', '{deeper_level}', '{lesson}',
            '{current_status}', '{future_direction}', '{quality}',
            '{past_memory}', '{present_situation}', '{change}',
            '{unchanged_core}', '{comparison}', '{aspect}', '{role}',
            '{感受}', '{时代}', '{characteristic1}', '{characteristic2}',
            '{characteristic3}', '{experience}', '{understanding}',
            '{treasure}', '{definition}', '{journey}', '{witness}',
            '{emotional_moment}', '{support}', '{bond}', '{limitation}',
            '{ultimate_meaning}', '{wish}', '{hope}', '{future_aspiration}',
            '{early_focus}'
        ]

        # 使用主题和相关词汇填充占位符
        topic_words = self._generate_topic_words(topic)

        for placeholder in placeholders:
            if placeholder in filled:
                replacement = random.choice(topic_words)
                filled = filled.replace(placeholder, replacement, 1)

        return filled

    def _generate_topic_words(self, topic):
        """根据主题生成相关词汇"""
        # 基础词库
        general_words = [
            "创新发展", "可持续增长", "深化改革", "优化升级", "协调推进",
            "质量提升", "效率优化", "模式创新", "体系完善", "机制健全",
            "水平提高", "能力增强", "结构优化", "布局合理", "功能完善",
            "服务优质", "管理规范", "运行高效", "保障有力", "支撑充分"
        ]

        return [topic] + general_words

    def generate_text(self, attribute, topic, target_length=1777):
        """生成指定长度的完整文本"""
        paragraphs = []
        current_length = 0

        # 生成段落直到达到目标长度
        while current_length < target_length * 0.9:  # 90%的目标长度
            paragraph = self.generate_paragraph(attribute, topic)
            paragraphs.append(paragraph)
            current_length += len(paragraph)

        text = '\n\n'.join(paragraphs)

        # 确保长度在范围内
        if len(text) < self.min_length:
            # 如果太短，继续添加段落
            while len(text) < self.min_length:
                paragraph = self.generate_paragraph(attribute, topic)
                text += '\n\n' + paragraph

        if len(text) > self.max_length:
            # 如果太长，截断到合适长度
            text = text[:self.max_length]
            # 找到最后一个句号
            last_period = text.rfind('。')
            if last_period > self.min_length:
                text = text[:last_period + 1]

        return text

    def generate_dataset(self, total_count=9000):
        """生成完整数据集"""
        print(f"\n开始生成 {total_count} 条人类风格文本...", flush=True)

        # 五种属性均衡分布
        attributes = ['议论', '说明', '描写', '记叙', '抒情']
        count_per_attr = total_count // len(attributes)

        # 主题列表（匹配AI数据的主题）
        topics = [
            "人工智能", "教育改革", "环境保护", "文化传承", "科技创新",
            "经济发展", "社会治理", "医疗健康", "城市规划", "数字经济",
            "气候变化", "可持续发展", "文化交流", "历史文化", "艺术创作",
            "体育运动", "饮食文化", "旅游发展", "建筑设计", "交通运输",
            "能源转型", "农业现代化", "工业升级", "金融创新", "法治建设",
            "国际合作", "区域发展", "创业创新", "人才培养", "公共服务",
            "社区建设", "家庭教育", "老龄化问题", "青年发展", "妇女权益",
            "儿童保护", "残疾人事业", "志愿服务", "慈善公益", "社会保障"
        ]

        texts = []
        for attr in attributes:
            print(f"  生成 {attr} 类文本: {count_per_attr} 条...", flush=True)

            for i in range(count_per_attr):
                topic = random.choice(topics)

                # 长度随机分布（模拟真实数据）
                # 大部分在平均值附近，少部分偏离
                if random.random() < 0.7:  # 70%在平均值±300
                    target_length = self.target_avg_length + random.randint(-300, 300)
                else:  # 30%更分散
                    target_length = random.randint(self.min_length, self.max_length)

                text = self.generate_text(attr, topic, target_length)

                texts.append({
                    'text': text,
                    'source': 'generated_human_style',
                    'attribute': attr,
                    'topic': topic,
                    'length': len(text),
                    'timestamp': datetime.now().isoformat()
                })

                if (i + 1) % 200 == 0:
                    print(f"    进度: {i+1}/{count_per_attr}", flush=True)

        print(f"\n✓ 生成完成！总计: {len(texts)} 条", flush=True)

        return pd.DataFrame(texts)


def main():
    print("=" * 60, flush=True)
    print("高质量人类风格文本生成程序", flush=True)
    print("=" * 60, flush=True)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    generator = QualityHumanTextGenerator()

    # 生成9000条文本
    df = generator.generate_dataset(total_count=9000)

    # 保存
    output_dir = "datasets/human_texts"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/human_style_texts_9000.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(f"\n✓ 已保存到: {filename}", flush=True)

    # 统计信息
    print("\n" + "=" * 60, flush=True)
    print("数据集统计", flush=True)
    print("=" * 60, flush=True)
    print(f"总计: {len(df)} 条", flush=True)
    print(f"平均长度: {df['length'].mean():.0f} 字符", flush=True)
    print(f"长度范围: {df['length'].min()} - {df['length'].max()}", flush=True)
    print(f"\n属性分布:", flush=True)
    print(df['attribute'].value_counts())

    print("\n" + "=" * 60, flush=True)
    print("下一步:", flush=True)
    print("  1. 合并AI文本和人类文本", flush=True)
    print("  2. 添加标签（AI=1, 人类=0）", flush=True)
    print("  3. 划分train/val/test集", flush=True)
    print("  4. 转换为BERT格式", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
