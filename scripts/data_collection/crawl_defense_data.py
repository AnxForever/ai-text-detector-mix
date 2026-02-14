#!/usr/bin/env python3
"""
答辩场景数据采集脚本

目标: 采集模型误判的Human文本类型，解决答辩演示问题

误判类型 (Human → AI):
1. 学术摘要 (100% AI) → 采集 CNKI/万方/百度学术
2. 地方新闻 (97.7% AI) → 采集 新华网/人民网/各省新闻
3. 考试通知 (97.2% AI) → 采集 大学官网教务通知
4. 学生邮件 (100% AI) → 模板生成 + 变体

用法:
    python scripts/data_collection/crawl_defense_data.py --type academic --count 500
    python scripts/data_collection/crawl_defense_data.py --type news --count 500
    python scripts/data_collection/crawl_defense_data.py --type notice --count 300
    python scripts/data_collection/crawl_defense_data.py --type all
"""
import os
import sys
import json
import time
import random
import argparse
import requests
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'defense_patch'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 请求配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

# 代理池 (可选)
PROXIES = None

def get_session():
    """创建请求会话"""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def clean_text(text):
    """清理文本"""
    if not text:
        return ""
    # 去除HTML标签残留
    text = re.sub(r'<[^>]+>', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    # 去除特殊字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 去除版权声明等
    text = re.sub(r'(责任编辑|来源：|作者：|编辑：)[^\n。]+', '', text)
    return text.strip()

def is_valid_text(text, min_len=80, max_len=800):
    """验证文本是否有效"""
    if not text or len(text) < min_len or len(text) > max_len:
        return False
    # 检查是否含有太多特殊字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < len(text) * 0.5:
        return False
    return True

def save_results(data, filename):
    """保存结果"""
    if not data:
        print(f"  无数据保存")
        return None

    output_path = OUTPUT_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  已保存 {len(data)} 条 → {output_path}")
    return output_path


# ============================================================
# 1. 学术摘要采集
# ============================================================

def crawl_baidu_xueshu(keywords, count_per_keyword=50):
    """
    从百度学术采集摘要
    https://xueshu.baidu.com
    """
    results = []
    session = get_session()

    for keyword in keywords:
        print(f"  采集关键词: {keyword}")
        collected = 0
        page = 0

        while collected < count_per_keyword and page < 10:
            try:
                url = f"https://xueshu.baidu.com/s?wd={quote(keyword)}&pn={page*10}&ie=utf-8"
                resp = session.get(url, timeout=15)

                if resp.status_code != 200:
                    print(f"    页面 {page} 请求失败: {resp.status_code}")
                    break

                soup = BeautifulSoup(resp.text, 'html.parser')

                # 提取摘要
                for item in soup.find_all('div', class_='sc_content'):
                    abstract_div = item.find('div', class_='c_abstract')
                    if not abstract_div:
                        continue

                    text = clean_text(abstract_div.get_text())
                    if is_valid_text(text, min_len=100, max_len=600):
                        results.append({
                            'text': text,
                            'label': 0,
                            'source': f'百度学术-{keyword}',
                            'category': 'academic',
                            'scenario': 'academic',
                            'collected_at': datetime.now().isoformat()
                        })
                        collected += 1

                        if collected >= count_per_keyword:
                            break

                page += 1
                time.sleep(random.uniform(1.5, 3.0))

            except Exception as e:
                print(f"    采集异常: {e}")
                break

        print(f"    {keyword}: 采集 {collected} 条")

    return results


def crawl_cnki_abstracts(keywords, count_per_keyword=30):
    """
    从CNKI采集摘要 (需要登录，这里用开放接口)
    """
    # CNKI需要登录，改用其他开放源
    # 这里提供一个框架，实际需要根据具体接口调整
    results = []
    print(f"  CNKI采集 (需要配置账号，暂时跳过)")
    return results


def collect_academic_data(target_count=500):
    """
    综合采集学术摘要
    """
    print("\n" + "=" * 60)
    print("采集学术摘要")
    print("=" * 60)

    keywords = [
        # 计算机
        "深度学习", "机器学习", "自然语言处理", "图像识别", "推荐系统",
        "知识图谱", "强化学习", "目标检测", "语义分割", "神经网络",
        # 医学
        "临床研究", "药物治疗", "诊断方法", "疗效观察", "病例分析",
        # 经济管理
        "企业管理", "财务分析", "市场营销", "供应链", "人力资源",
        # 教育
        "教学改革", "课程设计", "学习效果", "教育评价", "人才培养",
        # 社会科学
        "乡村振兴", "社会治理", "公共政策", "城市发展", "人口问题",
    ]

    count_per_keyword = max(target_count // len(keywords), 10)

    results = crawl_baidu_xueshu(keywords[:15], count_per_keyword)

    print(f"\n学术摘要总计: {len(results)} 条")
    save_results(results, 'academic_abstracts.jsonl')

    return results


# ============================================================
# 2. 新闻采集
# ============================================================

def crawl_xinhua_news(count=200):
    """
    从新华网采集新闻
    """
    results = []
    session = get_session()

    channels = [
        ("http://www.news.cn/politics/", "时政"),
        ("http://www.news.cn/fortune/", "财经"),
        ("http://www.news.cn/tech/", "科技"),
        ("http://www.news.cn/local/", "地方"),
        ("http://www.news.cn/edu/", "教育"),
    ]

    for base_url, channel_name in channels:
        print(f"  采集频道: {channel_name}")
        collected = 0

        try:
            resp = session.get(base_url, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 提取新闻链接
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/202' in href and href.endswith('.htm'):
                    full_url = urljoin(base_url, href)
                    if full_url not in links:
                        links.append(full_url)

            # 采集新闻内容
            for link in links[:count // len(channels)]:
                try:
                    article_resp = session.get(link, timeout=10)
                    if article_resp.status_code != 200:
                        continue

                    article_soup = BeautifulSoup(article_resp.text, 'html.parser')

                    # 提取正文
                    content_div = (
                        article_soup.find('div', id='detail') or
                        article_soup.find('div', class_='main-aticle') or
                        article_soup.find('div', class_='article')
                    )

                    if content_div:
                        paragraphs = content_div.find_all('p')
                        text = ' '.join([clean_text(p.get_text()) for p in paragraphs])

                        if is_valid_text(text, min_len=100, max_len=800):
                            results.append({
                                'text': text,
                                'label': 0,
                                'source': f'新华网-{channel_name}',
                                'category': 'news',
                                'scenario': 'news_report',
                                'url': link,
                                'collected_at': datetime.now().isoformat()
                            })
                            collected += 1

                    time.sleep(random.uniform(0.8, 1.5))

                except Exception as e:
                    continue

            print(f"    {channel_name}: 采集 {collected} 条")
            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"    {channel_name}: 采集失败 - {e}")

    return results


def crawl_people_news(count=200):
    """
    从人民网采集新闻
    """
    results = []
    session = get_session()

    channels = [
        ("http://politics.people.com.cn/", "时政"),
        ("http://finance.people.com.cn/", "财经"),
        ("http://scitech.people.com.cn/", "科技"),
        ("http://edu.people.com.cn/", "教育"),
    ]

    for base_url, channel_name in channels:
        print(f"  采集频道: 人民网-{channel_name}")

        try:
            resp = session.get(base_url, timeout=15)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 提取新闻链接
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/n1/202' in href or '/n2/202' in href:
                    if href.startswith('http'):
                        links.append(href)
                    else:
                        links.append(urljoin(base_url, href))

            collected = 0
            for link in links[:count // len(channels)]:
                try:
                    article_resp = session.get(link, timeout=10)
                    if article_resp.status_code != 200:
                        continue

                    article_soup = BeautifulSoup(article_resp.text, 'html.parser')
                    content_div = article_soup.find('div', class_='rm_txt_con')

                    if content_div:
                        paragraphs = content_div.find_all('p')
                        text = ' '.join([clean_text(p.get_text()) for p in paragraphs])

                        if is_valid_text(text, min_len=100, max_len=800):
                            results.append({
                                'text': text,
                                'label': 0,
                                'source': f'人民网-{channel_name}',
                                'category': 'news',
                                'scenario': 'news_report',
                                'url': link,
                                'collected_at': datetime.now().isoformat()
                            })
                            collected += 1

                    time.sleep(random.uniform(0.8, 1.5))

                except:
                    continue

            print(f"    人民网-{channel_name}: 采集 {collected} 条")

        except Exception as e:
            print(f"    人民网-{channel_name}: 采集失败 - {e}")

    return results


def collect_news_data(target_count=500):
    """
    综合采集新闻数据
    """
    print("\n" + "=" * 60)
    print("采集新闻报道")
    print("=" * 60)

    results = []

    # 新华网
    print("\n[新华网]")
    xinhua_data = crawl_xinhua_news(target_count // 2)
    results.extend(xinhua_data)

    # 人民网
    print("\n[人民网]")
    people_data = crawl_people_news(target_count // 2)
    results.extend(people_data)

    print(f"\n新闻总计: {len(results)} 条")
    save_results(results, 'news_reports.jsonl')

    return results


# ============================================================
# 3. 通知公告采集
# ============================================================

def crawl_university_notices(count=200):
    """
    从大学官网采集通知公告
    """
    results = []
    session = get_session()

    # 部分大学教务处/通知公告页面
    universities = [
        ("https://jwc.pku.edu.cn/tzgg/index.htm", "北京大学教务部"),
        ("https://info.tsinghua.edu.cn/", "清华大学信息门户"),
        ("https://jwc.fudan.edu.cn/tzgg/list.htm", "复旦大学教务处"),
        ("https://jwc.sjtu.edu.cn/xwzx/tztg.htm", "上海交大教务处"),
        ("https://jwc.zju.edu.cn/tzgg/list.htm", "浙江大学教务处"),
    ]

    for url, name in universities:
        print(f"  尝试采集: {name}")

        try:
            resp = session.get(url, timeout=15, verify=False)
            if resp.status_code != 200:
                print(f"    {name}: 请求失败 ({resp.status_code})")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 尝试提取通知列表
            # 不同大学页面结构不同，需要适配
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text().strip()
                # 通知一般以"关于"开头
                if text.startswith('关于') and len(text) > 10:
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    links.append((href, text))

            collected = 0
            for link, title in links[:count // len(universities)]:
                try:
                    article_resp = session.get(link, timeout=10, verify=False)
                    if article_resp.status_code != 200:
                        continue

                    article_soup = BeautifulSoup(article_resp.text, 'html.parser')

                    # 尝试多种内容选择器
                    content_div = (
                        article_soup.find('div', class_='wp_articlecontent') or
                        article_soup.find('div', class_='article-content') or
                        article_soup.find('div', class_='content') or
                        article_soup.find('div', id='vsb_content')
                    )

                    if content_div:
                        text = clean_text(content_div.get_text())

                        if is_valid_text(text, min_len=80, max_len=800):
                            results.append({
                                'text': text,
                                'label': 0,
                                'source': name,
                                'category': 'notice',
                                'scenario': 'formal_notice',
                                'title': title,
                                'url': link,
                                'collected_at': datetime.now().isoformat()
                            })
                            collected += 1

                    time.sleep(random.uniform(1, 2))

                except:
                    continue

            print(f"    {name}: 采集 {collected} 条")

        except Exception as e:
            print(f"    {name}: 采集失败 - {e}")

    return results


def generate_notice_templates(count=200):
    """
    生成通知模板变体
    基于真实通知格式，随机组合生成
    """
    results = []

    # 通知模板组件
    prefixes = [
        "各学院、各部门：",
        "全校师生：",
        "各位同学：",
        "各教学单位：",
        "全体教职工：",
        "各研究生培养单位：",
        "全体本科生：",
    ]

    topics = [
        ("根据学校教学工作安排，{time}将进行{event}。",
         [("2026年春季学期", "期中教学检查"), ("本学期第15周", "课程考核"), ("下周一至周五", "教学评估")]),
        ("为做好{event}工作，现将有关事项通知如下：",
         [("期末考试", ), ("毕业设计答辩", ), ("学位论文送审", ), ("新学期选课", )]),
        ("关于{event}的通知：经研究决定，{detail}。",
         [("调整上课时间", "自下周起实行新的作息时间表"),
          ("更换教室", "原在A楼上课的班级改至B楼"),
          ("补考安排", "补考时间定于开学第二周")]),
        ("为{purpose}，学校决定{action}。",
         [("加强校园安全管理", "对校门出入实行预约制度"),
          ("提高教学质量", "开展本学期教学督导工作"),
          ("丰富校园文化生活", "举办第十届校园文化艺术节")]),
    ]

    endings = [
        "请各单位认真组织落实，确保工作顺利进行。",
        "请相关人员按时参加，不得无故缺席。",
        "如有问题，请及时与相关部门联系。",
        "特此通知。",
        "请周知。",
        "望各单位遵照执行。",
    ]

    departments = [
        "教务处", "学生工作部", "研究生院", "保卫处", "后勤集团",
        "图书馆", "信息中心", "人事处", "科研处", "国际交流处"
    ]

    dates = [
        "2026年1月15日", "2026年2月20日", "2026年3月5日",
        "2025年12月30日", "2026年1月8日"
    ]

    for i in range(count):
        prefix = random.choice(prefixes)
        topic_template, options = random.choice(topics)
        option = random.choice(options)
        if len(option) == 1:
            topic = topic_template.format(event=option[0])
        elif len(option) == 2:
            topic = topic_template.format(time=option[0], event=option[1]) if 'time' in topic_template else \
                   topic_template.format(event=option[0], detail=option[1]) if 'detail' in topic_template else \
                   topic_template.format(purpose=option[0], action=option[1])

        ending = random.choice(endings)
        dept = random.choice(departments)
        date = random.choice(dates)

        text = f"{prefix}\n\n{topic}\n\n{ending}\n\n{dept}\n{date}"

        results.append({
            'text': text,
            'label': 0,
            'source': '模板生成-通知',
            'category': 'notice',
            'scenario': 'formal_notice',
            'collected_at': datetime.now().isoformat()
        })

    return results


def collect_notice_data(target_count=300):
    """
    综合采集通知公告
    """
    print("\n" + "=" * 60)
    print("采集通知公告")
    print("=" * 60)

    results = []

    # 大学官网采集
    print("\n[大学官网]")
    uni_data = crawl_university_notices(target_count // 2)
    results.extend(uni_data)

    # 模板生成
    print("\n[模板生成]")
    template_count = target_count - len(results)
    if template_count > 0:
        template_data = generate_notice_templates(template_count)
        results.extend(template_data)
        print(f"  生成模板通知: {len(template_data)} 条")

    print(f"\n通知公告总计: {len(results)} 条")
    save_results(results, 'formal_notices.jsonl')

    return results


# ============================================================
# 4. 日常分享采集 (微博/知乎)
# ============================================================

def crawl_weibo_public(count=300):
    """
    采集微博公开内容
    注: 微博反爬严格，需要Cookie或API
    """
    results = []
    print("  微博采集需要配置Cookie，暂时跳过")
    print("  建议: 手动导出或使用第三方数据集")
    return results


def generate_daily_sharing_templates(count=300):
    """
    生成日常分享模板
    基于真实日常分享格式
    """
    results = []

    templates = [
        # 天气+心情
        ("今天{weather}，{activity}，{feeling}。",
         [{"weather": "天气真好", "activity": "出去散了个步", "feeling": "心情舒畅"},
          {"weather": "下雨了", "activity": "在家看了一天书", "feeling": "很充实"},
          {"weather": "阳光明媚", "activity": "和朋友去爬山", "feeling": "累但开心"}]),

        # 工作日常
        ("今天{event}，{result}。{comment}",
         [{"event": "加班到很晚", "result": "终于把项目做完了", "comment": "累死了但是很有成就感。"},
          {"event": "开了一整天的会", "result": "脑子都是嗡嗡的", "comment": "回家只想躺着。"},
          {"event": "收到了offer", "result": "太开心了", "comment": "努力没有白费！"}]),

        # 美食分享
        ("今天{where}吃到了{food}，{taste}。{recommend}",
         [{"where": "在公司附近", "food": "一家新开的面馆", "taste": "味道很不错", "recommend": "推荐给大家！"},
          {"where": "朋友请客", "food": "一顿火锅", "taste": "辣得很过瘾", "recommend": "下次还想去。"},
          {"where": "自己在家做了", "food": "红烧肉", "taste": "第一次做居然成功了", "recommend": "成就感满满。"}]),

        # 购物评价
        ("买了{item}，{usage}，{evaluation}。{conclusion}",
         [{"item": "一个新耳机", "usage": "用了一周", "evaluation": "音质很好", "conclusion": "物超所值，好评！"},
          {"item": "网上推荐的护肤品", "usage": "试了几天", "evaluation": "感觉还不错", "conclusion": "会回购的。"},
          {"item": "新款手机", "usage": "体验了一下", "evaluation": "拍照效果很惊艳", "conclusion": "换机值了。"}]),

        # 学习分享
        ("最近在{learning}，{progress}。{thought}",
         [{"learning": "学Python", "progress": "终于能写点小程序了", "thought": "编程真的很有意思。"},
          {"learning": "准备考研", "progress": "复习到第三轮了", "thought": "希望能上岸！"},
          {"learning": "练习英语口语", "progress": "感觉进步挺大的", "thought": "坚持就是胜利。"}]),
    ]

    for i in range(count):
        template, options = random.choice(templates)
        option = random.choice(options)
        text = template.format(**option)

        results.append({
            'text': text,
            'label': 0,
            'source': '模板生成-日常分享',
            'category': 'daily',
            'scenario': 'daily_sharing',
            'collected_at': datetime.now().isoformat()
        })

    return results


def collect_daily_data(target_count=300):
    """
    综合采集日常分享
    """
    print("\n" + "=" * 60)
    print("采集日常分享")
    print("=" * 60)

    results = []

    # 微博采集 (暂时跳过)
    # weibo_data = crawl_weibo_public(target_count // 2)
    # results.extend(weibo_data)

    # 模板生成
    print("\n[模板生成]")
    template_data = generate_daily_sharing_templates(target_count)
    results.extend(template_data)
    print(f"  生成日常分享: {len(template_data)} 条")

    print(f"\n日常分享总计: {len(results)} 条")
    save_results(results, 'daily_sharing.jsonl')

    return results


# ============================================================
# 主函数
# ============================================================

def merge_all_data():
    """合并所有采集的数据"""
    print("\n" + "=" * 60)
    print("合并所有数据")
    print("=" * 60)

    all_data = []

    for jsonl_file in OUTPUT_DIR.glob('*.jsonl'):
        if jsonl_file.name == 'all_defense_patch.jsonl':
            continue
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        print(f"  加载 {jsonl_file.name}: {len(all_data)} 条")

    if all_data:
        # 去重
        seen = set()
        unique_data = []
        for item in all_data:
            key = item['text'][:50]
            if key not in seen:
                seen.add(key)
                unique_data.append(item)

        output_path = OUTPUT_DIR / 'all_defense_patch.jsonl'
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in unique_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"\n合并完成: {len(unique_data)} 条 (去重前 {len(all_data)} 条)")
        print(f"保存至: {output_path}")

        # 统计
        from collections import Counter
        cat_counts = Counter([d.get('category', 'unknown') for d in unique_data])
        print("\n按类别统计:")
        for cat, cnt in sorted(cat_counts.items()):
            print(f"  {cat}: {cnt}")

    return all_data


def main():
    parser = argparse.ArgumentParser(description='答辩场景数据采集')
    parser.add_argument('--type', choices=['academic', 'news', 'notice', 'daily', 'all'],
                       default='all', help='采集类型')
    parser.add_argument('--count', type=int, default=200, help='每类目标数量')
    args = parser.parse_args()

    print("=" * 60)
    print("答辩场景数据采集工具")
    print("=" * 60)
    print(f"采集类型: {args.type}")
    print(f"目标数量: {args.count}")
    print(f"输出目录: {OUTPUT_DIR}")

    if args.type in ['academic', 'all']:
        collect_academic_data(args.count)

    if args.type in ['news', 'all']:
        collect_news_data(args.count)

    if args.type in ['notice', 'all']:
        collect_notice_data(args.count)

    if args.type in ['daily', 'all']:
        collect_daily_data(args.count)

    if args.type == 'all':
        merge_all_data()

    print("\n" + "=" * 60)
    print("采集完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
