import openai
import pandas as pd
import time
import random
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

# ==================== 1. 配置多模型API客户端 ====================

MODEL_CONFIGS = {
    "custom": {  # 自定义API端点 (已测试可用)
        "client_class": openai.OpenAI,
        "api_key": "sk-p14OddEwPKmsWMVkBsmckJrKnMQRo8xSlzOhNcYmAtZ5JSbO",
        "base_url": "https://china.184772.xyz/v1",
        "model_name": "gpt-4o-mini"
    },
    "deepseek": {
        "client_class": openai.OpenAI,
        "api_key": "sk-c6d1be1eab4a4981b2efa3110f037376",
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "qwen": {  # 通义千问 (通过DashScope平台)
        "client_class": openai.OpenAI,
        "api_key": "sk-bfceae8516c94a5693e135d54fdc3900",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen3-max"
    },
    # 可以继续添加其他模型
}

# 初始化所有客户端
clients = {}
for model_name, config in MODEL_CONFIGS.items():
    try:
        clients[model_name] = config["client_class"](
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        print(f"✅ {model_name.upper()} 客户端初始化成功")
    except Exception as e:
        print(f"❌ {model_name.upper()} 客户端初始化失败: {e}")
        clients[model_name] = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 2. 自动维度生成器 ====================

class AutoDimensionGenerator:
    """使用LLM自动生成所有维度的内容"""

    def __init__(self, client, model="gpt-4o-mini"):
        self.client = client
        self.model = model
        self.cache_dir = "dimension_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_with_cache(self, cache_key: str, generation_func, *args, **kwargs):
        """带缓存的生成函数"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                if cached_data.get('expires', 0) > time.time():
                    logger.info(f"从缓存加载: {cache_key}")
                    return cached_data['data']

        # 生成新数据
        data = generation_func(*args, **kwargs)

        # 缓存24小时
        cache_data = {
            'data': data,
            'expires': time.time() + 24 * 3600,
            'created': datetime.now().isoformat()
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"生成并缓存: {cache_key}")
        return data

    def generate_topics(self, num_topics=30, categories=None):
        """自动生成多样化的话题"""
        if categories is None:
            categories = ["科技与创新", "社会与民生", "经济与商业", "教育与学习",
                          "健康与医疗", "环境与生态", "文化与艺术", "哲学与思考"]

        prompt = f"""请生成{num_topics}个具有深度讨论价值的写作话题。要求：

1. 话题应该：
   - 具有现实意义和讨论价值
   - 包含一定的争议性或多视角空间
   - 适合不同教育背景的人理解
   - 避免过于技术化的专业术语
   - 每个话题都是一个完整的句子

2. 话题类型分布：
   - 30% 当前社会热点问题
   - 30% 长期存在的根本性问题
   - 20% 未来趋势与预测性话题
   - 20% 跨学科交叉话题

3. 覆盖以下类别：{', '.join(categories)}

请直接返回话题列表，每个话题一行，不要编号。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()
            # 按行分割并清理
            topics = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            topics = [t for t in topics if len(t) > 5 and len(t) < 100]  # 长度过滤

            # 确保数量
            if len(topics) < num_topics:
                # 补充一些通用话题
                default_topics = [
                    "人工智能对创意产业的影响",
                    "全球化背景下的文化认同危机",
                    "社交媒体对青少年心理健康的影响",
                    "气候变化对农业生产的长远影响",
                    "远程办公对城市发展的改变",
                    "人口老龄化对社会福利的挑战",
                    "教育公平在数字化时代的实现路径",
                    "数据隐私与商业创新的平衡之道"
                ]
                topics.extend(default_topics[:num_topics - len(topics)])

            return topics[:num_topics]

        except Exception as e:
            logger.error(f"生成话题失败: {e}")
            return self._get_default_topics(num_topics)

    def generate_genres(self, num_genres=20):
        """自动生成多种文体格式"""
        prompt = f"""请生成{num_genres}种不同的写作体裁和格式，要求：

1. 覆盖各种实际应用场景：
   - 正式文档类（如报告、论文等）
   - 商业应用类（如邮件、提案等）
   - 媒体传播类（如文章、推文等）
   - 个人表达类（如日记、博客等）
   - 创意写作类（如小说、诗歌等）

2. 每种格式应有明确的特征描述，例如：
   "学术论文摘要（300-500字，需包含研究背景、方法、结果、结论）"
   "产品发布会演讲稿（富有感染力，突出产品亮点和用户价值）"

请直接返回格式列表，每个格式一行。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1500
            )

            content = response.choices[0].message.content.strip()
            genres = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            genres = [g for g in genres if len(g) > 3 and len(g) < 80]

            if len(genres) < num_genres:
                default_genres = [
                    "学术论文摘要",
                    "深度新闻报道",
                    "微信公众号文章",
                    "知乎高质量回答",
                    "个人博客文章",
                    "商业分析报告",
                    "产品测评报告",
                    "旅行游记",
                    "书评影评",
                    "日记片段"
                ]
                genres.extend(default_genres[:num_genres - len(genres)])

            return genres[:num_genres]

        except Exception as e:
            logger.error(f"生成文体失败: {e}")
            return self._get_default_genres(num_genres)

    def generate_roles(self, num_roles=15):
        """自动生成多样化的角色视角"""
        prompt = f"""请生成{num_roles}个不同的写作角色和视角，要求：

1. 多样化覆盖：
   - 不同年龄阶段（学生、青年、中年、老年）
   - 不同职业背景（专业人士、自由职业者、企业家等）
   - 不同立场态度（支持者、反对者、中立者、怀疑者等）
   - 不同生活经历（亲历者、观察者、研究者、受影响者等）

2. 每个角色应有独特的视角特征，例如：
   "一名关注科技伦理的哲学家"
   "经历过数字化转型的传统行业从业者"
   "对新技术既期待又担忧的普通消费者"

请直接返回角色列表，每个角色一行。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1500
            )

            content = response.choices[0].message.content.strip()
            roles = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            roles = [r for r in roles if len(r) > 3 and len(r) < 50]

            if len(roles) < num_roles:
                default_roles = [
                    "一名高中生",
                    "大学相关专业教授",
                    "行业分析师",
                    "创业者",
                    "资深记者",
                    "普通消费者",
                    "退休老人",
                    "亲历者",
                    "持怀疑态度的观察者",
                    "乐观的支持者"
                ]
                roles.extend(default_roles[:num_roles - len(roles)])

            return roles[:num_roles]

        except Exception as e:
            logger.error(f"生成角色失败: {e}")
            return self._get_default_roles(num_roles)

    def generate_styles(self, num_styles=10):
        """自动生成写作风格"""
        prompt = f"""请生成{num_styles}种不同的写作风格描述，要求：

每种风格应包含语言特点和情感基调，例如：
"严谨学术风格：逻辑严密，引用数据，客观中立"
"热情洋溢风格：富有感染力，使用修辞手法，情感充沛"

请直接返回风格列表，每个风格一行。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()
            styles = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            styles = [s for s in styles if len(s) > 5 and len(s) < 50]

            if len(styles) < num_styles:
                default_styles = [
                    "简洁客观，平实直述",
                    "严谨学术，引用数据与文献",
                    "热情洋溢，富有感染力",
                    "幽默风趣，带点调侃",
                    "娓娓道来，充满故事性",
                    "批判性思维，多角度质疑",
                    "诗意化表达，注重意象",
                    "对话式语气，亲切自然"
                ]
                styles.extend(default_styles[:num_styles - len(styles)])

            return styles[:num_styles]

        except Exception as e:
            logger.error(f"生成风格失败: {e}")
            return self._get_default_styles(num_styles)

    def generate_constraints(self, num_constraints=8):
        """自动生成写作约束"""
        prompt = f"""请生成{num_constraints}个不同的写作要求或约束条件，要求：

每个约束应具体明确，具有可操作性，例如：
"全文不少于400字"
"至少包含三个分论点或三个核心细节"
"引用一个真实的数据、研究或历史事件"

请直接返回约束列表，每个约束一行。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )

            content = response.choices[0].message.content.strip()
            constraints = [line.strip(' "-') for line in content.split('\n') if line.strip()]
            constraints = [c for c in constraints if len(c) > 5 and len(c) < 60]

            if len(constraints) < num_constraints:
                default_constraints = [
                    "全文不少于400字",
                    "至少包含三个分论点或三个核心细节",
                    "引用一个真实的数据、研究或历史事件",
                    "避免使用任何专业术语，力求通俗易懂",
                    "在结尾处提出一个引人深思的问题",
                    "使用一个生动的比喻或类比来说明核心观点",
                    "包含正反两方面的观点分析",
                    "提供具体的行动建议或解决方案"
                ]
                constraints.extend(default_constraints[:num_constraints - len(constraints)])

            return constraints[:num_constraints]

        except Exception as e:
            logger.error(f"生成约束失败: {e}")
            return self._get_default_constraints(num_constraints)

    def _get_default_topics(self, num):
        """默认话题"""
        topics = [
            "人工智能在医疗诊断中的最新突破",
            "量子计算机能否破解当前的所有加密算法",
            "在线教育平台如何保障学生的学习效果与专注度",
            "一二线城市年轻人为何兴起'反向消费'趋势",
            "社区养老与机构养老各自的优势与挑战",
            "正念冥想对缓解职场人焦虑情绪的有效性研究",
            "代糖饮料是否真的比含糖饮料更健康",
            "极端天气频发，个人如何参与气候变化应对",
            "城市'海绵化'改造对解决内涝问题的实际效果",
            "网红经济对传统消费品营销模式的冲击与启示",
            "小微企业如何在数字经济时代找到生存空间"
        ]
        return topics[:min(num, len(topics))]

    def _get_default_genres(self, num):
        """默认文体"""
        genres = [
            "博客文章",
            "学术论文摘要",
            "产品测评报告",
            "正式的商业邮件",
            "微信公众号推文",
            "知乎问答",
            "小学生作文",
            "日记片段"
        ]
        return genres[:min(num, len(genres))]

    def _get_default_roles(self, num):
        """默认角色"""
        roles = [
            "一名中学生",
            "大学相关专业教授",
            "行业分析师",
            "对此话题持乐观态度的支持者",
            "持谨慎怀疑态度的观察者",
            "事件亲历者或受影响者"
        ]
        return roles[:min(num, len(roles))]

    def _get_default_styles(self, num):
        """默认风格"""
        styles = [
            "简洁客观，平实直述",
            "严谨学术，引用数据与文献",
            "热情洋溢，富有感染力",
            "幽默风趣，带点调侃",
            "娓娓道来，充满故事性",
            "批判性思维，多角度质疑"
        ]
        return styles[:min(num, len(styles))]

    def _get_default_constraints(self, num):
        """默认约束"""
        constraints = [
            "全文不少于400字",
            "至少包含三个分论点或三个核心细节",
            "引用一个真实的数据、研究或历史事件",
            "避免使用任何专业术语，力求通俗易懂",
            "在结尾处提出一个引人深思的问题",
            "使用一个生动的比喻或类比来说明核心观点"
        ]
        return constraints[:min(num, len(constraints))]


# ==================== 3. 智能组合生成器 ====================

class IntelligentCombinationGenerator:
    """智能生成合理的六维组合"""

    def __init__(self, auto_generator):
        self.auto_gen = auto_generator

    def generate_combinations(self, num_combinations=1000, attributes=None):
        """生成智能组合"""
        if attributes is None:
            attributes = ["描写", "记叙", "说明", "抒情", "议论"]

        # 生成所有维度
        logger.info("开始生成维度内容...")

        # 使用缓存机制
        topics = self.auto_gen.generate_with_cache(
            f"topics_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_topics,
            num_topics=50
        )

        genres = self.auto_gen.generate_with_cache(
            f"genres_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_genres,
            num_genres=20
        )

        roles = self.auto_gen.generate_with_cache(
            f"roles_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_roles,
            num_roles=15
        )

        styles = self.auto_gen.generate_with_cache(
            f"styles_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_styles,
            num_styles=10
        )

        constraints = self.auto_gen.generate_with_cache(
            f"constraints_{datetime.now().strftime('%Y%m%d')}",
            self.auto_gen.generate_constraints,
            num_constraints=8
        )

        logger.info(f"维度生成完成: 话题{len(topics)}个, 文体{len(genres)}个, "
                    f"角色{len(roles)}个, 风格{len(styles)}个, 约束{len(constraints)}个")

        combinations = []

        for i in range(num_combinations):
            # 智能选择：确保合理匹配
            attribute = random.choice(attributes)
            topic = random.choice(topics)

            # 根据话题选择合适角色
            role = self._select_appropriate_role(topic, roles)

            # 根据话题和角色选择合适文体
            genre = self._select_appropriate_genre(topic, role, genres)

            # 根据文体选择合适风格
            style = self._select_appropriate_style(genre, styles)

            # 随机选择约束
            constraint = random.choice(constraints)

            # 生成提示词
            prompt = self._generate_prompt(attribute, topic, genre, role, style, constraint)

            # 计算质量评分
            quality_score = self._estimate_quality(attribute, topic, genre, role, style, constraint)

            combinations.append({
                "plan_id": i,
                "attribute": attribute,
                "topic": topic,
                "genre": genre,
                "role": role,
                "style": style,
                "constraint": constraint,
                "prompt": prompt,
                "quality_score": quality_score,
                "generated": False  # 标记是否已生成
            })

            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{num_combinations} 个组合")

        # 按质量评分排序
        combinations.sort(key=lambda x: x["quality_score"], reverse=True)

        return combinations

    def _select_appropriate_role(self, topic, roles):
        """根据话题选择合适的角色"""
        # 简单关键词匹配
        topic_lower = topic.lower()

        # 科技话题适合技术相关角色
        tech_keywords = ["人工", "智能", "量子", "计算", "数据", "算法", "科技", "技术", "数字"]
        if any(kw in topic_lower for kw in tech_keywords):
            tech_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                  ["技术", "科学", "工程", "研究", "开发"])]
            if tech_roles:
                return random.choice(tech_roles)

        # 社会话题适合普通公众角色
        social_keywords = ["社会", "民生", "生活", "社区", "家庭", "消费", "养老"]
        if any(kw in topic_lower for kw in social_keywords):
            social_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                    ["市民", "居民", "消费者", "老人", "青年"])]
            if social_roles:
                return random.choice(social_roles)

        # 经济话题适合商业相关角色
        economic_keywords = ["经济", "商业", "市场", "金融", "创业", "投资", "营销"]
        if any(kw in topic_lower for kw in economic_keywords):
            economic_roles = [r for r in roles if any(kw in r.lower() for kw in
                                                      ["分析", "企业", "商业", "投资", "市场"])]
            if economic_roles:
                return random.choice(economic_roles)

        # 默认随机选择
        return random.choice(roles)

    def _select_appropriate_genre(self, topic, role, genres):
        """根据话题和角色选择合适的文体"""
        role_lower = role.lower()

        # 学生角色适合教育类文体
        if any(kw in role_lower for kw in ["学生", "孩子", "青少年"]):
            educational_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                           ["作文", "日记", "读后感", "周记"])]
            if educational_genres:
                return random.choice(educational_genres)

        # 专业角色适合专业文体
        if any(kw in role_lower for kw in ["教授", "专家", "分析", "研究", "工程师"]):
            professional_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                            ["论文", "报告", "分析", "研究", "学术"])]
            if professional_genres:
                return random.choice(professional_genres)

        # 普通公众角色适合媒体文体
        if any(kw in role_lower for kw in ["消费者", "市民", "用户", "读者", "亲历者"]):
            media_genres = [g for g in genres if any(kw in g.lower() for kw in
                                                     ["博客", "文章", "评论", "问答", "推文"])]
            if media_genres:
                return random.choice(media_genres)

        # 默认随机选择
        return random.choice(genres)

    def _select_appropriate_style(self, genre, styles):
        """根据文体选择合适的风格"""
        genre_lower = genre.lower()

        # 正式文体适合严谨风格
        if any(kw in genre_lower for kw in ["学术", "报告", "论文", "正式", "商业", "分析"]):
            formal_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                      ["严谨", "客观", "学术", "正式", "批判"])]
            if formal_styles:
                return random.choice(formal_styles)

        # 媒体文体适合活泼风格
        if any(kw in genre_lower for kw in ["微信", "博客", "日记", "游记", "故事", "推文"]):
            informal_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                        ["幽默", "热情", "故事", "亲切", "娓娓道来"])]
            if informal_styles:
                return random.choice(informal_styles)

        # 创意文体适合文学风格
        if any(kw in genre_lower for kw in ["诗歌", "小说", "散文", "创意"]):
            creative_styles = [s for s in styles if any(kw in s.lower() for kw in
                                                        ["诗意", "故事", "抒情", "感染力"])]
            if creative_styles:
                return random.choice(creative_styles)

        # 默认随机选择
        return random.choice(styles)

    def _generate_prompt(self, attribute, topic, genre, role, style, constraint):
        """生成六维提示词"""
        attribute_instructions = {
            "描写": f"请对以下主题进行具体、细腻的**描写**，聚焦于感官细节和画面营造：",
            "记叙": f"请围绕以下主题，**记叙**一个完整的事件或故事，讲清来龙去脉：",
            "说明": f"请对以下主题进行客观、清晰、有条理的**说明**，解释其事实或原理：",
            "抒情": f"请就以下主题，**抒发**你真实的情感、感受或思考：",
            "议论": f"请就以下主题，发表明确的观点并进行有力的**议论**和论证："
        }

        parts = []
        parts.append(f"【任务】{attribute_instructions.get(attribute, '')}")
        parts.append(f"【主题】{topic}")
        parts.append(f"【你的角色】请以{role}的身份进行写作。")
        parts.append(f"【文体与发布平台】请写成一篇{genre}。")
        parts.append(f"【语言风格】整体文风请保持{style}。")
        parts.append(f"【特殊要求】{constraint}。")

        final_prompt = "\n".join(parts)
        final_prompt += "\n\n请综合以上所有要求，创作一篇完整、连贯的文章。"
        return final_prompt

    def _estimate_quality(self, attribute, topic, genre, role, style, constraint):
        """估计组合的质量分数（0-1）"""
        score = 0.5  # 基础分

        # 话题质量加分
        if len(topic) > 10 and len(topic) < 80:
            score += 0.1

        # 角色与话题匹配度加分
        if self._check_compatibility(role, topic):
            score += 0.2

        # 文体与风格匹配度加分
        if self._check_genre_style_match(genre, style):
            score += 0.1

        # 避免过于复杂的组合
        if len(constraint) < 50:
            score += 0.05

        # 多样性加分
        score += random.uniform(0, 0.05)

        return min(score, 1.0)

    def _check_compatibility(self, role, topic):
        """检查角色与话题的兼容性"""
        tech_keywords = ["科技", "AI", "人工", "智能", "数据", "算法", "量子", "数字"]
        if any(kw in topic for kw in tech_keywords):
            if any(kw in role for kw in ["技术", "科学", "工程", "研究", "开发"]):
                return True

        social_keywords = ["社会", "民生", "生活", "社区", "文化", "家庭"]
        if any(kw in topic for kw in social_keywords):
            if any(kw in role for kw in ["市民", "居民", "观察者", "亲历者", "消费者"]):
                return True

        return False

    def _check_genre_style_match(self, genre, style):
        """检查文体与风格的匹配度"""
        formal_keywords = ["学术", "报告", "论文", "正式", "商业", "分析"]
        if any(kw in genre for kw in formal_keywords):
            if any(kw in style for kw in ["严谨", "客观", "学术", "正式", "批判"]):
                return True

        informal_keywords = ["微信", "博客", "日记", "游记", "故事", "推文"]
        if any(kw in genre for kw in informal_keywords):
            if any(kw in style for kw in ["幽默", "热情", "故事", "亲切", "娓娓道来"]):
                return True

        return False


# ==================== 4. 增强型API调用器 ====================

def generate_text_with_retry(prompt: str, target_model: str, max_retries: int = 3) -> Optional[str]:
    """调用指定模型生成文本，失败时自动重试"""
    if target_model not in clients or clients[target_model] is None:
        print(f"    ⚠️  模型 {target_model} 未配置，跳过")
        return None

    config = MODEL_CONFIGS[target_model]
    for attempt in range(max_retries):
        try:
            response = clients[target_model].chat.completions.create(
                model=config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
                stream=False
            )
            return response.choices[0].message.content.strip()

        except openai.APIError as e:
            wait_time = (attempt + 1) * 5  # 指数退避
            print(
                f"    ⚠️  {target_model.upper()} API错误 (尝试 {attempt + 1}/{max_retries}): {e}. {wait_time}秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    ❌  {target_model.upper()} 非预期错误: {e}")
            return None

    print(f"    ❌  {target_model.upper()} 在{max_retries}次重试后仍失败")
    return None


# ==================== 5. 智能质量评估器 ====================

class TextQualityAssessor:
    """评估生成文本的质量"""

    def __init__(self, client):
        self.client = client

    def assess_quality(self, text, prompt_info):
        """评估文本质量"""
        if not text or len(text.strip()) < 50:
            return 0.0

        # 基础质量检查
        score = 0.5

        # 长度检查
        expected_min_length = 300 if "不少于400字" in prompt_info.get("constraint", "") else 100
        if len(text) >= expected_min_length:
            score += 0.2

        # 结构检查
        if '\n\n' in text or text.count('。') >= 3:
            score += 0.1

        # 内容检查
        if "三个分论点" in prompt_info.get("constraint", ""):
            if text.count('首先') + text.count('其次') + text.count('再次') + text.count('第一') + text.count(
                    '第二') + text.count('第三') >= 2:
                score += 0.1

        if "比喻" in prompt_info.get("constraint", "") or "类比" in prompt_info.get("constraint", ""):
            if "像" in text or "如同" in text or "好比" in text:
                score += 0.1

        # 避免AI模板回复
        if "作为一个AI" not in text and "很抱歉" not in text and "无法完成" not in text:
            score += 0.1

        return min(score, 1.0)


# ==================== 6. 主控制器：完全自动化生成 ====================

def main_auto():
    """完全自动化的数据集生成系统"""
    print("=" * 70)
    print("          完全自动化六维文本数据集生成系统")
    print("=" * 70)

    # 检查可用的客户端
    active_clients = {k: v for k, v in clients.items() if v is not None}
    if not active_clients:
        print("❌ 没有可用的API客户端")
        return

    print(f"✅ 可用的模型：{list(active_clients.keys())}")

    # 配置生成参数
    SAMPLES_PER_MODEL = 800  # 每个模型生成800条数据
    ACTIVE_MODELS = list(active_clients.keys())  # 使用所有可用模型
    OUTPUT_DIR = "auto_generated_datasets"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化组件（优先使用custom客户端）
    primary_client = active_clients.get("custom") or active_clients.get("deepseek") or list(active_clients.values())[0]
    auto_gen = AutoDimensionGenerator(primary_client)
    comb_gen = IntelligentCombinationGenerator(auto_gen)
    quality_assessor = TextQualityAssessor(primary_client)

    # 生成组合计划
    plan_file = os.path.join(OUTPUT_DIR, "auto_generation_plan.json")

    if os.path.exists(plan_file):
        logger.info("检测到已有生成计划，将从中断处继续...")
        with open(plan_file, 'r', encoding='utf-8') as f:
            generation_plan = json.load(f)
    else:
        logger.info("创建新的生成计划...")
        generation_plan = comb_gen.generate_combinations(
            num_combinations=SAMPLES_PER_MODEL,
            attributes=["描写", "记叙", "说明", "抒情", "议论"]
        )

        # 保存计划
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(generation_plan, f, ensure_ascii=False, indent=2)

        logger.info(f"生成计划已保存，共{len(generation_plan)}条任务。")
        print(f"📊 生成计划质量分布：")
        scores = [item["quality_score"] for item in generation_plan]
        print(f"   平均质量: {sum(scores) / len(scores):.2f}")
        print(f"   最高质量: {max(scores):.2f}")
        print(f"   最低质量: {min(scores):.2f}")

    # 为每个模型生成数据
    for model_name in ACTIVE_MODELS:
        if model_name not in active_clients:
            print(f"\n⏭️  跳过模型 {model_name.upper()} (配置无效)")
            continue

        print(f"\n{'=' * 50}")
        print(f"开始为模型 {model_name.upper()} 生成数据")
        print(f"{'=' * 50}")

        output_file = os.path.join(OUTPUT_DIR,
                                   f"auto_dataset_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        data_records = []
        failed_count = 0
        success_count = 0
        total_quality = 0

        # 只处理未生成的高质量组合
        ungenerated_plans = [p for p in generation_plan if not p.get(f"generated_{model_name}", False)]

        # 按质量排序，先生成高质量的组合
        ungenerated_plans.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        for idx, plan_item in enumerate(ungenerated_plans[:SAMPLES_PER_MODEL]):
            print(f"[{model_name.upper()} {idx + 1}/{min(len(ungenerated_plans), SAMPLES_PER_MODEL)}] "
                  f"质量:{plan_item['quality_score']:.2f} 属性:{plan_item['attribute']} 主题:{plan_item['topic'][:15]}...")

            text = generate_text_with_retry(plan_item["prompt"], model_name)
            time.sleep(0.5)  # 控制请求频率

            if text:
                # 评估质量
                quality = quality_assessor.assess_quality(text, plan_item)
                total_quality += quality
                success_count += 1

                record = {
                    "text_id": f"{model_name}_{plan_item['plan_id']}",
                    "text_content": text,
                    "source_model": model_name,
                    "attribute": plan_item['attribute'],
                    "topic": plan_item['topic'],
                    "genre": plan_item['genre'],
                    "role": plan_item['role'],
                    "style": plan_item['style'],
                    "constraint": plan_item['constraint'],
                    "prompt": plan_item['prompt'],
                    "combination_quality": plan_item['quality_score'],
                    "generation_quality": quality,
                    "length": len(text),
                    "timestamp": datetime.now().isoformat()
                }
                data_records.append(record)
                plan_item[f"generated_{model_name}"] = True  # 标记为已生成

                quality_str = f"质量:{quality:.2f}"
                if quality > 0.8:
                    quality_str = f"✅ 优秀 {quality_str}"
                elif quality > 0.6:
                    quality_str = f"✓ 良好 {quality_str}"
                else:
                    quality_str = f"⚠️ 一般 {quality_str}"

                print(f"    {quality_str} ({len(text)} 字符)")
            else:
                failed_count += 1
                print(f"    ❌ 失败")

            # 每生成20条或结束时，保存一次进度
            if (idx + 1) % 20 == 0 or (idx + 1) == min(len(ungenerated_plans), SAMPLES_PER_MODEL):
                # 保存数据
                if data_records:
                    df = pd.DataFrame(data_records)
                    # 如果是追加模式（非首次保存）
                    if os.path.exists(output_file) and idx >= 20:
                        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                    else:
                        df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    data_records = []  # 清空内存

                # 更新生成计划（保存进度）
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(generation_plan, f, ensure_ascii=False, indent=2)
                print(f"    💾 进度已保存")

        avg_quality = total_quality / success_count if success_count > 0 else 0
        print(f"\n📊 {model_name.upper()} 生成完成！")
        print(f"   成功: {success_count}, 失败: {failed_count}")
        print(f"   平均生成质量: {avg_quality:.2f}")
        print(f"📁 数据文件: {output_file}")

    # 最终合并所有模型的数据
    print(f"\n{'=' * 70}")
    print("正在合并所有模型的数据集...")
    all_data_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("auto_dataset_") and f.endswith(".csv")]
    combined_df = pd.DataFrame()

    for file in all_data_files:
        try:
            df = pd.read_csv(os.path.join(OUTPUT_DIR, file), encoding='utf-8-sig')
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            logger.error(f"读取文件 {file} 失败: {e}")

    if not combined_df.empty:
        combined_file = os.path.join(OUTPUT_DIR,
                                     f"AUTO_COMBINED_DATASET_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_df.to_csv(combined_file, index=False, encoding='utf-8-sig')

        print(f"✅ 合并完成！总计 {len(combined_df)} 条数据。")
        print(f"📁 合并文件: {combined_file}")

        # 打印统计信息
        print(f"\n📈 数据质量统计:")
        print(f"   平均组合质量: {combined_df['combination_quality'].mean():.2f}")
        print(f"   平均生成质量: {combined_df['generation_quality'].mean():.2f}")

        print(f"\n📊 数据分布统计:")
        print("1. 按模型来源分布:")
        print(combined_df['source_model'].value_counts())

        print("\n2. 按文本属性分布:")
        print(combined_df['attribute'].value_counts())

        print("\n3. 话题多样性:")
        print(f"   唯一话题数: {combined_df['topic'].nunique()}")

        print("\n4. 文本长度统计:")
        print(f"   平均长度: {combined_df['length'].mean():.0f} 字符")
        print(f"   最短: {combined_df['length'].min()} 字符")
        print(f"   最长: {combined_df['length'].max()} 字符")

        # 保存元数据
        meta_file = os.path.join(OUTPUT_DIR, f"meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        meta_data = {
            "generation_time": datetime.now().isoformat(),
            "total_samples": len(combined_df),
            "models_used": list(combined_df['source_model'].unique()),
            "avg_combination_quality": float(combined_df['combination_quality'].mean()),
            "avg_generation_quality": float(combined_df['generation_quality'].mean()),
            "dimension_stats": {
                "attributes": combined_df['attribute'].value_counts().to_dict(),
                "topics_count": int(combined_df['topic'].nunique()),
                "genres": combined_df['genre'].value_counts().to_dict(),
                "roles": combined_df['role'].value_counts().to_dict()
            },
            "text_length_stats": {
                "avg": float(combined_df['length'].mean()),
                "min": int(combined_df['length'].min()),
                "max": int(combined_df['length'].max())
            }
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)

        print(f"\n📋 元数据保存至: {meta_file}")

        # 质量分布可视化
        quality_bins = pd.cut(combined_df['generation_quality'],
                              bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0],
                              labels=['很差', '较差', '一般', '良好', '优秀'])
        print(f"\n📊 生成质量分布:")
        print(quality_bins.value_counts().sort_index())

    else:
        print("❌ 未找到可合并的数据文件。")


# ==================== 7. 程序入口 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("          文本数据集生成系统")
    print("=" * 70)
    print("请选择生成模式：")
    print("1. 原始版本（预设维度）")
    print("2. 自动版本（AI生成维度，推荐）")

    choice = input("请输入选择（1或2）：").strip()

    if choice == "1":
        # 注意：原始版本需要导入原始代码中的main函数
        # 由于原始代码已经被修改，这里提供一个兼容方式
        print("\n⚠️  注意：原始版本需要单独运行原代码文件")
        print("正在启动自动化版本...")
        main_auto()
    else:
        main_auto()