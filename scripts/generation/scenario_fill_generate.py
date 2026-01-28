#!/usr/bin/env python3
"""Generate scenario-based AI samples from a JSON task config."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.utils.api_config import get_proxy_config


FORBIDDEN_PHRASES = [
    "作为AI",
    "作为一个AI",
    "作为人工智能",
    "我无法",
    "我不能",
    "无法回答",
    "无法提供",
]

LENGTH_BUCKET_RANGES: Dict[str, Tuple[int, int]] = {
    "80-200": (80, 350),      # 放宽：短文本难控制，允许到350
    "200-500": (160, 720),    # 放宽20%
    "500-1000": (360, 1320),  # 放宽20%
    "1000-2000": (720, 2640), # 放宽20%
    "2000+": (1440, 4080),    # 放宽20%
}

# 截断目标长度（用于智能截断）
LENGTH_BUCKET_TRUNCATE_TARGET: Dict[str, int] = {
    "80-200": 300,
    "200-500": 650,
    "500-1000": 1200,
    "1000-2000": 2400,
    "2000+": 3800,
}

LENGTH_BUCKET_HINTS: Dict[str, str] = {
    "80-200": "总字数约 120-200 字",
    "200-500": "总字数约 300-450 字",
    "500-1000": "总字数约 650-900 字",
    "1000-2000": "总字数约 1200-1700 字",
    "2000+": "总字数约 2200-3200 字",
}

MAX_TOKENS_BY_BUCKET: Dict[str, int] = {
    "80-200": 240,
    "200-500": 300,
    "500-1000": 800,
    "1000-2000": 1200,
    "2000+": 1600,
}

LIST_PATTERN = re.compile(
    r"(^\\s*[-*•])|(^\\s*\\d+[\\.)、])|(^\\s*[一二三四五六七八九十]+、)",
    re.MULTILINE,
)
CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_SPLIT = re.compile(r"([。！？!?])")


def chinese_ratio(text: str) -> float:
    """Compute rough Chinese character ratio."""
    if not text:
        return 0.0
    chinese = len(CHINESE_CHAR.findall(text))
    return chinese / max(len(text), 1)


def clean_forbidden_phrases(text: str) -> Tuple[str, bool]:
    """Remove lines/sentences containing forbidden phrases."""
    if not text or not any(phrase in text for phrase in FORBIDDEN_PHRASES):
        return text, False

    lines = [line for line in text.splitlines() if not any(p in line for p in FORBIDDEN_PHRASES)]
    cleaned = "\n".join(lines).strip() if lines else text

    parts = SENTENCE_SPLIT.split(cleaned)
    kept: List[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if any(p in sentence for p in FORBIDDEN_PHRASES):
            continue
        kept.append(sentence + punct)
    cleaned = "".join(kept).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if not cleaned:
        return text, False
    return cleaned, cleaned != text


def smart_truncate(text: str, target_len: int, min_len: int = 50) -> Tuple[str, bool]:
    """
    Smart truncation: cut text at sentence boundary to fit target length.

    Args:
        text: Input text
        target_len: Target maximum length
        min_len: Minimum acceptable length after truncation

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if len(text) <= target_len:
        return text, False

    # Split into sentences
    parts = SENTENCE_SPLIT.split(text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if sentence.strip():
            sentences.append(sentence + punct)

    # Build text up to target length
    result_parts: List[str] = []
    current_len = 0
    for sent in sentences:
        if current_len + len(sent) <= target_len:
            result_parts.append(sent)
            current_len += len(sent)
        else:
            # Try to include partial sentence if it's the first one
            if not result_parts and len(sent) > min_len:
                # Find last sentence boundary within limit
                for i in range(min(len(sent), target_len) - 1, min_len, -1):
                    if sent[i] in "。！？!?.":
                        result_parts.append(sent[: i + 1])
                        break
            break

    if not result_parts:
        # Fallback: hard cut at target_len, try to end at punctuation
        for i in range(min(len(text), target_len) - 1, max(min_len, target_len // 2), -1):
            if text[i] in "。！？!?.,;；，":
                return text[: i + 1], True
        return text[:target_len], True

    truncated = "".join(result_parts).strip()
    if len(truncated) < min_len:
        return text, False  # Don't truncate if result too short

    return truncated, True


@dataclass
class DecodingProfile:
    """Decoding profile for generation."""

    name: str
    temperature: float
    top_p: float
    max_tokens: int


@dataclass
class Task:
    """Single generation task."""

    scenario_id: str
    scenario: str
    style_plan: str
    length_bucket: str
    target_count: int
    priority: str
    domain_risk: str = "normal"
    hs_domain: str = ""
    ood_split: str = ""


SCENARIO_META = {
    "A": {
        "name": "学业写作",
        "audience": "本科生",
        "tone": "语气正式、学术化",
        "extra": "贴近课程/实验/论文写作场景",
    },
    "B": {
        "name": "职场写作",
        "audience": "团队成员",
        "tone": "语气正式、偏工作文档",
        "extra": "包含行动项或检查清单风格",
    },
    "C": {
        "name": "公共知识与科普问答",
        "audience": "普通读者",
        "tone": "客观解释、通俗但不口水",
        "extra": "突出概念与注意事项",
    },
    "D": {
        "name": "社区/社媒短文本",
        "audience": "社区用户",
        "tone": "可适度口语化，但不使用极端情绪或粗口",
        "extra": "像经验帖或求助帖的要点整理",
    },
    "E": {
        "name": "消费与产品内容",
        "audience": "消费者",
        "tone": "客观中立、偏评测/选购建议",
        "extra": "强调参数、优缺点或使用场景",
    },
    "F": {
        "name": "资讯/新闻风",
        "audience": "公众",
        "tone": "客观中立、资讯摘要风格",
        "extra": "强调关键事实与条目式信息",
    },
}

SCENARIO_TOPICS = {
    "A": [
        # 理工实验
        "电路实验报告要点",
        "化学滴定实验记录",
        "力学实验数据分析",
        "微生物培养实验总结",
        "光学干涉实验记录",
        # 课程作业
        "统计学课程作业复盘",
        "软件工程课程项目总结",
        "数据库设计课程作业",
        "操作系统课程实验报告",
        "计算机网络课程笔记",
        # 论文写作
        "生态学课程论文框架",
        "文献综述写作方法",
        "毕业论文开题报告要点",
        "学术论文摘要撰写技巧",
        "实验方法章节写作规范",
        # 学习方法
        "考研复习计划安排",
        "英语四六级备考策略",
        "高等数学学习笔记",
        "编程入门自学路线",
        "期末复习重点整理",
        # 文科人文
        "历史事件因果分析",
        "哲学概念辨析笔记",
        "社会调查报告框架",
        "教育心理学课堂笔记",
        "法学案例分析要点",
    ],
    "B": [
        # 周报月报
        "产品上线周报",
        "研发团队月度总结",
        "运维值班周报",
        "市场推广效果月报",
        "客服团队工作周报",
        # 故障处理
        "故障复盘行动项",
        "线上事故处理报告",
        "系统异常排查记录",
        "数据库故障恢复流程",
        "网络中断应急处置",
        # 流程文档
        "版本发布前检查清单",
        "客户问题处理流程",
        "新员工入职培训大纲",
        "代码审查规范文档",
        "测试用例编写指南",
        # 项目管理
        "月度目标执行清单",
        "项目风险评估报告",
        "需求变更审批流程",
        "迭代回顾会议纪要",
        "跨部门协作沟通方案",
        # 行政人事
        "年度绩效考核标准",
        "出差报销流程说明",
        "办公设备采购申请",
        "团建活动策划方案",
        "会议室预约使用规范",
    ],
    "C": [
        # 自然科学
        "光合作用的核心流程",
        "地球板块运动原理",
        "基因编辑技术简介",
        "太阳能发电工作原理",
        "气候变化的主要原因",
        # 健康医疗
        "血压监测的注意事项",
        "睡眠质量提升方法",
        "常见过敏症状识别",
        "疫苗接种基本常识",
        "心肺复苏急救步骤",
        # 生活常识
        "垃圾分类的常见误区",
        "家庭用电安全须知",
        "食品保质期正确理解",
        "饮用水质量判断方法",
        "家居防潮防霉技巧",
        # 科技概念
        "区块链的关键概念",
        "人工智能发展简史",
        "5G通信技术特点",
        "云计算与边缘计算区别",
        "物联网应用场景介绍",
        # 社会人文
        "个人信息保护要点",
        "碳中和概念与路径",
        "数字货币基础知识",
        "知识产权保护常识",
        "心理健康自我评估",
    ],
    "D": [
        # 租房生活
        "租房避坑经验整理",
        "合租室友相处指南",
        "搬家打包省钱攻略",
        "租房合同注意事项",
        "水电燃气缴费指南",
        # 兴趣爱好
        "新手摄影入门清单",
        "健身打卡经验帖",
        "家庭烘焙入门指南",
        "露营装备选购经验",
        "养猫新手避坑记录",
        # 消费维权
        "网购维权步骤",
        "快递丢件理赔流程",
        "二手交易防骗技巧",
        "退换货政策解读",
        "消费投诉渠道汇总",
        # 通勤出行
        "通勤省钱技巧",
        "地铁换乘路线规划",
        "自驾出行准备清单",
        "高铁抢票实用技巧",
        "骑行通勤装备推荐",
        # 职场社交
        "面试准备经验分享",
        "简历优化技巧帖",
        "职场新人生存指南",
        "副业收入经验交流",
        "社保公积金查询方法",
    ],
    "E": [
        # 数码产品
        "手机选购对比要点",
        "笔记本电脑对比指标",
        "耳机试听注意事项",
        "平板电脑选购建议",
        "智能手表功能对比",
        # 家电家居
        "空气净化器选型指南",
        "咖啡机购买清单",
        "扫地机器人性能对比",
        "净水器滤芯更换指南",
        "智能门锁选购要点",
        # 美妆个护
        "防晒产品成分解读",
        "洗面奶选择建议",
        "电动牙刷使用体验",
        "护发产品功效对比",
        "香水入门选购指南",
        # 食品饮料
        "婴儿奶粉成分对比",
        "即食食品营养分析",
        "咖啡豆产地风味区别",
        "功能饮料成分解读",
        "零食热量排行参考",
        # 服务评测
        "宽带套餐资费对比",
        "云存储服务横评",
        "在线教育平台对比",
        "外卖平台满减攻略",
        "快递公司时效对比",
    ],
    "F": [
        # 产品更新
        "版本更新公告",
        "产品功能更新要点",
        "APP重大改版说明",
        "系统升级注意事项",
        "新功能上线预告",
        # 政策法规
        "政策发布摘要",
        "新规实施要点解读",
        "行业标准更新通知",
        "税收政策调整简报",
        "教育改革方案概览",
        # 行业资讯
        "行业动态简报",
        "科技企业季度财报摘要",
        "市场份额变化分析",
        "融资上市动态汇总",
        "行业峰会要点回顾",
        # 活动通知
        "活动发布通知",
        "线下展会参展指南",
        "线上直播预告",
        "社区活动报名须知",
        "年度评选结果公示",
        # 事件报道
        "自然灾害预警通报",
        "交通管制通知摘要",
        "公共卫生事件简报",
        "重大赛事赛程速报",
        "文化节庆活动安排",
    ],
}

PLACEHOLDER_VALUES: Dict[str, List[str]] = {
    "课程/实验主题": [
        "文本分类实验", "图像识别实验", "数据结构课程作业",
        "计算机组成原理实验", "信号与系统仿真", "编译原理课程设计",
        "人工智能导论实验", "嵌入式系统开发", "网络安全攻防实验",
    ],
    "概念/方法名": [
        "梯度下降", "支持向量机", "注意力机制",
        "卷积神经网络", "决策树", "K近邻算法",
        "主成分分析", "长短期记忆网络", "Transformer架构",
    ],
    "对比概念": [
        "随机梯度下降", "朴素贝叶斯", "循环神经网络",
        "逻辑回归", "随机森林", "生成对抗网络",
        "变分自编码器", "图神经网络", "迁移学习",
    ],
    "研究问题": [
        "AI 文本检测的鲁棒性提升", "多域文本分类泛化", "模型校准与阈值选择",
        "小样本学习的有效策略", "数据增强对模型性能的影响", "跨语言迁移的可行性",
        "对抗样本的防御方法", "模型可解释性提升", "联邦学习隐私保护",
    ],
    "分类/检测/对比": ["检测", "分类", "对比", "识别", "预测", "评估"],
    "系统/功能": [
        "权限管理", "日志分析", "订单检索",
        "消息推送", "文件上传", "用户认证",
        "数据导出", "报表生成", "搜索引擎",
    ],
    "负责人": ["李明", "王珊", "张磊", "陈晨", "刘洋", "赵欣", "孙鹏", "周文"],
    "联系方式": [
        "li@example.com", "wang@example.com", "zhang@example.com",
        "chen@example.com", "liu@example.com", "zhao@example.com",
    ],
    "项目名": [
        "星河项目", "海豚计划", "北斗升级",
        "天眼工程", "鸿蒙迭代", "银河计划",
        "凤凰重构", "麒麟优化", "昆仑扩展",
    ],
    "故障/延期/质量问题": [
        "服务超时", "上线延期", "质量回归",
        "内存泄漏", "接口异常", "数据丢失",
        "性能下降", "兼容性问题", "安全漏洞",
    ],
    "日期": [
        "2026-02-01", "2026-02-15", "2026-03-01",
        "2026-03-15", "2026-04-01", "2026-04-15",
    ],
    "CHG-xxxx": ["CHG-1024", "CHG-2048", "CHG-3096", "CHG-4012", "CHG-5108", "CHG-6200"],
    "产品名": [
        "云笔记", "智能手环", "语音助手",
        "智慧屏", "运动相机", "阅读器",
        "翻译笔", "学习平板", "智能音箱",
    ],
    "型号": ["X100", "Pro-2", "Lite-3", "Max-5", "Air-4", "SE-2", "Ultra-1", "Mini-6"],
    "商品名": [
        "无线耳机", "便携投影", "扫地机器人",
        "电动牙刷", "空气炸锅", "筋膜枪",
        "充电宝", "降噪耳机", "智能台灯",
    ],
    "场景": ["通勤", "居家", "学习", "出差", "运动", "办公", "旅行", "户外"],
    "对比商品/旧款": [
        "上一代型号", "同价位竞品", "旧款产品",
        "海外版本", "国产替代品", "高端对标款",
    ],
    "对比产品": [
        "竞品A", "竞品B", "上一代产品",
        "同系列高配版", "国际品牌同类产品", "平价替代方案",
    ],
    "工具/概念/服务": [
        "账号管理", "云同步", "权限审批",
        "在线协作", "版本控制", "数据备份",
        "工单系统", "知识库", "自动化部署",
    ],
    "话题": [
        "政策变化", "产品体验", "服务质量",
        "环保议题", "教育改革", "科技伦理",
        "城市规划", "医疗保障", "数据隐私",
    ],
    "对方观点一句话": [
        "价格不值", "功能不实用", "更新太频繁",
        "售后差", "稳定性不够", "学习成本高",
    ],
    "背景要点3条": [
        "预算有限；设备老旧；时间紧迫",
        "团队新手多；文档不全；需求变更频繁",
        "用户量激增；服务器满载；交付日期临近",
    ],
    "问题主题": [
        "网速不稳定", "功能冲突", "账号异常",
        "支付失败", "数据同步延迟", "权限不足",
        "页面加载慢", "通知不及时", "兼容性报错",
    ],
    "事件主题": [
        "产品更新", "行业调整", "政策发布",
        "技术突破", "安全事件", "市场变化",
    ],
    "事件/产品/功能": [
        "新功能上线", "产品改版", "活动规则",
        "会员体系升级", "界面全新设计", "API接口开放",
    ],
    "专题主题": [
        "生成式 AI 监管", "内容平台治理", "数据合规实践",
        "个人信息保护法落地", "算法推荐透明化", "数字版权管理",
    ],
    "社区主题": [
        "摄影交流", "职场经验", "生活互助",
        "读书分享", "宠物养护", "DIY手工",
        "健身打卡", "美食探店", "旅行攻略",
    ],
    "科普主题": [
        "睡眠与健康", "人工智能基础", "垃圾分类",
        "疫苗原理", "气候变化", "太空探索",
        "营养均衡", "心理调适", "急救常识",
    ],
    "高中生/非技术人员/产品经理": ["高中生", "非技术人员", "产品经理", "大学新生", "退休人员", "自由职业者"],
    "支持/反对/中立偏质疑": ["支持", "反对", "中立偏质疑", "观望", "有条件支持"],
    "普通公众/从业者": ["普通公众", "从业者", "学生", "家长", "管理者"],
    "vX.Y.Z": ["v2.4.1", "v3.0.0", "v1.8.5", "v4.1.0", "v2.0.3", "v5.2.1"],
    "姓名": ["李明", "王珊", "张磊", "陈晨", "刘洋", "赵欣", "孙鹏", "周文", "吴静", "郑凯"],
}


def render_template(template: str, values: Dict[str, str], fallback: str) -> str:
    """Safely render a template with fallback replacements."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            return values[key]
        if key in PLACEHOLDER_VALUES:
            return random.choice(PLACEHOLDER_VALUES[key])
        if "/" in key:
            options = [item.strip() for item in key.split("/") if item.strip()]
            if options:
                return random.choice(options)
        return fallback

    return re.sub(r"\{([^{}]+)\}", replace, template)

PROMPT_LIBRARY = {
    "list": [
        {
            "id": "LIST-01",
            "template": (
                "你要为{audience}编写{scenario_name}场景的要点清单。主题：{topic}。"
                "请用编号或条目列出 4-6 条要点，每条不超过20字。"
                "{tone}。{extra}。{length_hint}。禁止自我指代和免责声明。"
                "不要输出需求分析/格式说明/标题，只输出正文，每条尽量简短。"
            ),
        },
        {
            "id": "LIST-02",
            "template": (
                "生成一份{scenario_name}相关的检查清单，主题：{topic}，面向{audience}。"
                "使用条目符号或“1. 2. 3.”格式，列出4-6条。"
                "{tone}。{extra}。{length_hint}。"
                "不要解释任务要求，只输出清单正文。"
            ),
        },
    ],
    "guide": [
        {
            "id": "GUIDE-01",
            "template": (
                "请撰写{scenario_name}场景的操作指南，主题：{topic}，面向{audience}。"
                "结构必须包含：目标/准备、步骤（至少5步）、注意事项（至少3条）。"
                "{tone}。{extra}。{length_hint}。禁止自我指代。"
                "不要拆解需求，不要写“说明/要求”，只输出正文。"
            ),
        },
        {
            "id": "GUIDE-02",
            "template": (
                "编写一份{scenario_name}使用说明，主题：{topic}。"
                "要求：分小节，含步骤与注意事项，条理清晰。"
                "面向{audience}。{tone}。{extra}。{length_hint}。"
                "不要包含分析过程或标题，只输出正文。"
            ),
        },
    ],
}

EXTERNAL_PROMPT_LIBRARY: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
EXTERNAL_STYLE_LIBRARY: Dict[str, List[Dict[str, str]]] = {}


def load_external_templates(path: Path) -> List[Dict[str, str]]:
    """Load a simple YAML template list (no external dependency)."""
    templates: List[Dict[str, str]] = []
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")
    text = path.read_text(encoding="utf-8")
    blocks = text.split("\n- id:")
    if not blocks:
        return templates
    for block in blocks[1:]:
        block = "id:" + block
        lines = block.splitlines()
        item: Dict[str, str] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("id:"):
                item["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("  scenario_id:"):
                item["scenario_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("  style_plan:"):
                item["style_plan"] = line.split(":", 1)[1].strip()
            elif line.startswith("  length_bucket:"):
                item["length_bucket"] = line.split(":", 1)[1].strip()
            elif line.startswith("  notes:"):
                item["notes"] = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("  template: |"):
                i += 1
                template_lines: List[str] = []
                while i < len(lines):
                    content = lines[i]
                    if content.startswith("    "):
                        template_lines.append(content[4:])
                    elif content.strip() == "":
                        template_lines.append("")
                    else:
                        break
                    i += 1
                item["template"] = "\n".join(template_lines).rstrip()
                continue
            i += 1
        if item.get("id") and item.get("template"):
            templates.append(item)
    return templates


def build_prompt_library_from_templates(
    templates: Iterable[Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str, str], List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """Index templates by (scenario_id, style_plan, length_bucket) and by style."""
    by_key: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    by_style: Dict[str, List[Dict[str, str]]] = {}
    for tmpl in templates:
        scenario_id = tmpl.get("scenario_id", "")
        style_plan = tmpl.get("style_plan", "")
        length_bucket = tmpl.get("length_bucket", "")
        key = (scenario_id, style_plan, length_bucket)
        by_key.setdefault(key, []).append(tmpl)
        by_style.setdefault(style_plan, []).append(tmpl)
    return by_key, by_style


def load_config(path: Path) -> Dict[str, Any]:
    """Load JSON config file."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_decoding_profiles(data: Iterable[Dict[str, Any]]) -> List[DecodingProfile]:
    """Parse decoding profiles from config."""
    profiles: List[DecodingProfile] = []
    for item in data:
        profiles.append(
            DecodingProfile(
                name=item["name"],
                temperature=float(item["temperature"]),
                top_p=float(item["top_p"]),
                max_tokens=int(item["max_tokens"]),
            )
        )
    if not profiles:
        raise ValueError("No decoding profiles configured.")
    return profiles


def parse_tasks(data: Iterable[Dict[str, Any]]) -> List[Task]:
    """Parse task list from config."""
    tasks: List[Task] = []
    for item in data:
        tasks.append(
            Task(
                scenario_id=item["scenario_id"],
                scenario=item["scenario"],
                style_plan=item["style_plan"],
                length_bucket=item["length_bucket"],
                target_count=int(item["target_count"]),
                priority=item.get("priority", ""),
                domain_risk=item.get("domain_risk", "normal"),
                hs_domain=item.get("hs_domain", ""),
                ood_split=item.get("ood_split", ""),
            )
        )
    if not tasks:
        raise ValueError("No tasks configured.")
    return tasks


def build_prompt(task: Task) -> Tuple[str, str, str, str]:
    """Build a prompt for the given task."""
    scenario_meta = SCENARIO_META[task.scenario_id]
    topic = random.choice(SCENARIO_TOPICS[task.scenario_id])
    template_notes = ""

    if EXTERNAL_PROMPT_LIBRARY:
        key = (task.scenario_id, task.style_plan, task.length_bucket)
        candidates = EXTERNAL_PROMPT_LIBRARY.get(key)
        if not candidates:
            candidates = EXTERNAL_STYLE_LIBRARY.get(task.style_plan, [])
        if not candidates:
            raise ValueError(
                f"No template for scenario={task.scenario_id} style={task.style_plan} length={task.length_bucket}"
            )
        template = random.choice(candidates)
        template_notes = template.get("notes", "")
    else:
        template = random.choice(PROMPT_LIBRARY[task.style_plan])

    length_hint = LENGTH_BUCKET_HINTS.get(task.length_bucket, "字数适中")
    values = {
        "scenario_name": scenario_meta["name"],
        "audience": scenario_meta["audience"],
        "tone": scenario_meta["tone"],
        "extra": scenario_meta["extra"],
        "topic": topic,
        "length_hint": length_hint,
    }
    if EXTERNAL_PROMPT_LIBRARY:
        prompt = render_template(template["template"], values, topic)
    else:
        prompt = template["template"].format(**values)
    return template["id"], topic, prompt, template_notes


def validate_text(text: str, style_plan: str, length_bucket: str) -> Tuple[bool, str]:
    """Validate generated text against constraints."""
    if not text:
        return False, "empty"
    if any(phrase in text for phrase in FORBIDDEN_PHRASES):
        return False, "forbidden_phrase"
    if chinese_ratio(text) < 0.3:
        return False, "low_chinese_ratio"
    min_len, max_len = LENGTH_BUCKET_RANGES.get(length_bucket, (0, 10_000))
    if not (min_len <= len(text) <= max_len):
        return False, "length_out_of_range"
    if style_plan == "list":
        has_marker = bool(LIST_PATTERN.search(text))
        has_lines = text.count("\n") >= 3
        has_hint = ("清单" in text) or ("要点" in text)
        if not (has_marker or has_lines or has_hint):
            return False, "missing_list_marker"
    if style_plan == "guide" and ("步骤" not in text and "注意" not in text):
        return False, "missing_guide_sections"
    return True, "ok"


def select_proxy(proxy_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick a proxy entry for this request."""
    return random.choice(proxy_entries)


def select_model(proxy: Dict[str, Any]) -> str:
    """Pick a model from proxy."""
    return random.choice(proxy["models"])


def select_decoding(decoding_profiles: List[DecodingProfile]) -> DecodingProfile:
    """Pick decoding profile."""
    return random.choice(decoding_profiles)


def call_api(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    decoding: DecodingProfile,
    timeout: float,
    max_tokens: int,
) -> Optional[str]:
    """Call chat completion API and return text."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是中文写作助手，只输出最终正文，不要分析、不要列出要求、不使用Markdown标题。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning_content") or ""
        return content.strip() if content else None
    except Exception:
        return None


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a JSONL record to file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_models(api_base: str, api_key: str) -> List[str]:
    """Fetch model ids from /models."""
    url = f"{api_base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json().get("data", [])
        models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        return models
    except Exception:
        return []


def filter_models(
    models: List[str],
    include: List[str],
    exclude: List[str],
    max_models: int,
) -> List[str]:
    """Filter model ids by include/exclude substrings."""
    filtered: List[str] = []
    for model in models:
        lower = model.lower()
        if include and not any(token in lower for token in include):
            continue
        if exclude and any(token in lower for token in exclude):
            continue
        filtered.append(model)
    if max_models and max_models > 0:
        return filtered[:max_models]
    return filtered


def trim_list_text(text: str, max_len: int) -> str:
    """Trim list-style text to fit max length by dropping trailing lines."""
    if len(text) <= max_len:
        return text
    lines = [line for line in text.splitlines() if line.strip()]
    kept: List[str] = []
    for line in lines:
        candidate = "\n".join(kept + [line])
        if len(candidate) > max_len:
            break
        kept.append(line)
    if len(kept) >= 3:
        return "\n".join(kept)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Scenario-based AI generation runner.")
    parser.add_argument(
        "--config",
        default="configs/scenario_fill_p0_2026-01-27.json",
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Limit per task for smoke tests (0 = no limit).",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    template_path = config.get("template_path")
    if template_path:
        resolved_path = Path(template_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(PROJECT_ROOT) / resolved_path
        templates = load_external_templates(resolved_path)
        global EXTERNAL_PROMPT_LIBRARY, EXTERNAL_STYLE_LIBRARY
        EXTERNAL_PROMPT_LIBRARY, EXTERNAL_STYLE_LIBRARY = build_prompt_library_from_templates(
            templates
        )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_pool = config.get("proxy_pool")
    proxy_entries: List[Dict[str, Any]] = []
    default_request_delay = float(config.get("request_delay", 1.0))
    if proxy_pool:
        for item in proxy_pool:
            proxy_name = item.get("proxy_name", "local_proxy")
            proxy_env_prefix = item.get("proxy_env_prefix", "LOCAL_PROXY")
            proxy_base_url = item.get("proxy_base_url")
            api_cfg = get_proxy_config(proxy_name, proxy_env_prefix, proxy_base_url)
            api_url = f"{api_cfg['url'].rstrip('/')}/chat/completions"
            models: List[str] = item.get("models", [])
            if item.get("models_from_api"):
                include = [str(x).lower() for x in item.get("model_include", [])]
                exclude = [str(x).lower() for x in item.get("model_exclude", [])]
                max_models = int(item.get("max_models", 0))
                api_models = fetch_models(api_cfg["url"], api_cfg["key"])
                filtered = filter_models(api_models, include, exclude, max_models)
                if filtered:
                    models = filtered
            if not models:
                continue
            proxy_entries.append(
                {
                    "proxy_name": proxy_name,
                    "api_key": api_cfg["key"],
                    "api_url": api_url,
                    "models": models,
                    "request_delay": float(item.get("request_delay", default_request_delay)),
                }
            )
        if not proxy_entries:
            raise ValueError("No proxy entries configured or discovered.")
    else:
        proxy_name = config.get("proxy_name", "local_proxy")
        proxy_env_prefix = config.get("proxy_env_prefix", "LOCAL_PROXY")
        proxy_base_url = config.get("proxy_base_url")
        api_cfg = get_proxy_config(proxy_name, proxy_env_prefix, proxy_base_url)
        api_url = f"{api_cfg['url'].rstrip('/')}/chat/completions"

        models = config.get("models", [])
        if config.get("models_from_api"):
            include = [str(x).lower() for x in config.get("model_include", [])]
            exclude = [str(x).lower() for x in config.get("model_exclude", [])]
            max_models = int(config.get("max_models", 0))
            api_models = fetch_models(api_cfg["url"], api_cfg["key"])
            filtered = filter_models(api_models, include, exclude, max_models)
            if filtered:
                models = filtered
        if not models:
            raise ValueError("No models configured or discovered.")
        proxy_entries = [
            {
                "proxy_name": proxy_name,
                "api_key": api_cfg["key"],
                "api_url": api_url,
                "models": models,
                "request_delay": default_request_delay,
            }
        ]
    decoding_profiles = parse_decoding_profiles(config["decoding_profiles"])
    tasks = parse_tasks(config["tasks"])

    request_delay = default_request_delay
    max_attempts = int(config.get("max_attempts", 3))
    timeout = float(config.get("request_timeout", 60))
    concurrency = int(config.get("concurrency", 1))
    request_interval = float(config.get("request_interval", request_delay))
    clean_forbidden = bool(config.get("clean_forbidden_phrases", False))
    if concurrency < 1:
        concurrency = 1

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_size = int(config.get("chunk_size", 50))
    log_every = int(config.get("log_every", 10))
    file_index = 1
    out_path = output_dir / f"ai_scenario_fill_{run_id}_part{file_index:03d}.jsonl"
    summary_path = output_dir / f"ai_scenario_fill_{run_id}_summary.json"
    rejected_path = output_dir / f"ai_scenario_fill_{run_id}_rejected.jsonl"

    success_total = 0
    failed_total = 0
    success_in_file = 0
    file_counts: Dict[str, int] = {}

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "tasks": [],
        "success_total": 0,
        "failed_total": 0,
    }

    task_queue: "queue.Queue[int]" = queue.Queue()
    task_summaries: List[Dict[str, Any]] = []
    total_target = 0
    for idx, task in enumerate(tasks):
        target = task.target_count
        if args.limit_per_task and args.limit_per_task > 0:
            target = min(target, args.limit_per_task)
        total_target += target
        task_summaries.append(
            {
                "scenario": task.scenario,
                "scenario_id": task.scenario_id,
                "style_plan": task.style_plan,
                "length_bucket": task.length_bucket,
                "target": target,
                "success": 0,
                "failed": 0,
            }
        )
        for _ in range(target):
            task_queue.put(idx)

    summary["total_target"] = total_target
    summary["concurrency"] = concurrency
    summary["request_interval"] = request_interval
    summary["clean_forbidden_phrases"] = clean_forbidden

    state_lock = threading.Lock()
    proxy_rate_lock = {proxy["proxy_name"]: threading.Lock() for proxy in proxy_entries}
    proxy_last_time = {proxy["proxy_name"]: 0.0 for proxy in proxy_entries}

    print(
        f"[start] run_id={run_id} total_target={total_target} "
        f"concurrency={concurrency} request_interval={request_interval}",
        flush=True,
    )

    def worker(worker_id: int) -> None:
        nonlocal success_total, failed_total, success_in_file, file_index, out_path, file_counts
        while True:
            try:
                task_index = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                task = tasks[task_index]
                success = False
                for _attempt in range(max_attempts):
                    prompt_id, topic, prompt, template_notes = build_prompt(task)
                    proxy = select_proxy(proxy_entries)
                    model = select_model(proxy)
                    decoding = select_decoding(decoding_profiles)
                    bucket_max = MAX_TOKENS_BY_BUCKET.get(task.length_bucket, decoding.max_tokens)
                    max_tokens = min(decoding.max_tokens, bucket_max)

                    rate_key = proxy["proxy_name"]
                    min_interval = max(
                        request_interval,
                        float(proxy.get("request_delay", request_delay)),
                    )
                    with proxy_rate_lock[rate_key]:
                        now = time.time()
                        elapsed = now - proxy_last_time[rate_key]
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                        proxy_last_time[rate_key] = time.time()

                    text = call_api(
                        proxy["api_url"],
                        proxy["api_key"],
                        model,
                        prompt,
                        decoding,
                        timeout,
                        max_tokens,
                    )
                    if text:
                        if task.style_plan == "list":
                            max_len = LENGTH_BUCKET_RANGES.get(
                                task.length_bucket,
                                (0, len(text)),
                            )[1]
                            text = trim_list_text(text, max_len)
                        cleaned_text = text
                        cleaned_applied = False
                        truncated_applied = False
                        if clean_forbidden:
                            cleaned_text, cleaned_applied = clean_forbidden_phrases(text)
                        # Apply smart truncation for overly long text
                        target_len = LENGTH_BUCKET_TRUNCATE_TARGET.get(task.length_bucket)
                        max_len = LENGTH_BUCKET_RANGES.get(task.length_bucket, (0, 99999))[1]
                        if target_len and len(cleaned_text) > max_len:
                            cleaned_text, truncated_applied = smart_truncate(
                                cleaned_text, target_len, min_len=80
                            )
                        valid, reason = validate_text(
                            cleaned_text,
                            task.style_plan,
                            task.length_bucket,
                        )
                        if valid:
                            text_id = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()[:12]
                            spec_like = 1 if "spec_like=1" in (template_notes or "") else 0
                            record = {
                                "text_id": text_id,
                                "text": cleaned_text,
                                "label": 1,
                                "scenario": task.scenario,
                                "scenario_id": task.scenario_id,
                                "style_plan": task.style_plan,
                                "answer_type": task.style_plan,
                                "domain_risk": task.domain_risk,
                                "hs_domain": task.hs_domain,
                                "ood_split": task.ood_split,
                                "length_bucket": task.length_bucket,
                                "model": model,
                                "proxy_name": proxy["proxy_name"],
                                "prompt_id": prompt_id,
                                "template_notes": template_notes,
                                "spec_like": spec_like,
                                "topic": topic,
                                "decoding_profile": decoding.name,
                                "temperature": decoding.temperature,
                                "top_p": decoding.top_p,
                                "max_tokens": max_tokens,
                                "source_type": "ai_generated",
                                "constraints_version": config.get("constraints_version", "v2"),
                                "created_at": datetime.now().isoformat(),
                                "cleaned_forbidden": int(cleaned_applied),
                                "truncated": int(truncated_applied),
                            }
                            with state_lock:
                                append_jsonl(out_path, record)
                                success_total += 1
                                task_summaries[task_index]["success"] += 1
                                success_in_file += 1
                                file_counts[model] = file_counts.get(model, 0) + 1
                                if success_total % log_every == 0:
                                    pending = total_target - (success_total + failed_total)
                                    print(
                                        f"[progress] total={success_total} failed={failed_total} "
                                        f"pending={pending} file={out_path.name} "
                                        f"last_model={model} proxy={proxy['proxy_name']}",
                                        flush=True,
                                    )
                                if success_in_file >= chunk_size:
                                    chunk_summary_path = output_dir / (
                                        f"ai_scenario_fill_{run_id}_part{file_index:03d}_summary.json"
                                    )
                                    chunk_summary = {
                                        "run_id": run_id,
                                        "part": file_index,
                                        "records": success_in_file,
                                        "models": file_counts,
                                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    }
                                    with chunk_summary_path.open("w", encoding="utf-8") as f:
                                        json.dump(chunk_summary, f, ensure_ascii=False, indent=2)
                                    print(
                                        f"[rotate] part={file_index:03d} records={success_in_file} "
                                        f"next_file_start",
                                        flush=True,
                                    )
                                    file_index += 1
                                    out_path = output_dir / (
                                        f"ai_scenario_fill_{run_id}_part{file_index:03d}.jsonl"
                                    )
                                    success_in_file = 0
                                    file_counts = {}
                            success = True
                            break
                        rejected = {
                            "scenario": task.scenario,
                            "scenario_id": task.scenario_id,
                            "style_plan": task.style_plan,
                            "answer_type": task.style_plan,
                            "domain_risk": task.domain_risk,
                            "hs_domain": task.hs_domain,
                            "ood_split": task.ood_split,
                            "length_bucket": task.length_bucket,
                            "model": model,
                            "prompt_id": prompt_id,
                            "template_notes": template_notes,
                            "topic": topic,
                            "decoding_profile": decoding.name,
                            "reason": reason,
                            "text": cleaned_text,
                            "cleaned_forbidden": int(cleaned_applied),
                        }
                        with state_lock:
                            append_jsonl(rejected_path, rejected)
                    continue
                if not success:
                    with state_lock:
                        task_summaries[task_index]["failed"] += 1
                        failed_total += 1
            finally:
                task_queue.task_done()

    threads = []
    for i in range(concurrency):
        thread = threading.Thread(target=worker, args=(i,), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    summary["tasks"] = task_summaries
    summary["success_total"] = success_total
    summary["failed_total"] = failed_total
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Output: {out_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
