#!/usr/bin/env python3
"""
从本地数据集提取答辩场景数据 + 新华网在线采集

策略: 优先从已有数据集提取，爬虫作为补充
数据源:
  - LCSTS (10k新闻正文) → 新闻/学术类
  - THUCNews (10k新闻) → 新闻类
  - 新华网 → 在线采集新闻
  - 模板生成 → 通知/日常分享

用法:
    python scripts/data_collection/extract_defense_data.py --all
    python scripts/data_collection/extract_defense_data.py --source lcsts --count 500
    python scripts/data_collection/extract_defense_data.py --source thucnews --count 500
    python scripts/data_collection/extract_defense_data.py --source xinhua --count 200
    python scripts/data_collection/extract_defense_data.py --source templates --count 300
    python scripts/data_collection/extract_defense_data.py --merge
"""
import json
import random
import re
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from collections import Counter
from urllib.parse import urljoin

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'datasets' / 'defense_patch' / 'extracted'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """清理文本"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'(责任编辑|来源：|作者：|编辑：)[^\n。]{0,30}', '', text)
    # 去掉LCSTS的任务前缀
    text = re.sub(r'^在本任务中，[^。]+。', '', text)
    return text.strip()


def is_valid_chinese(text, min_len=80, max_len=600):
    """检查是否为有效中文文本"""
    if not text or len(text) < min_len or len(text) > max_len:
        return False
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    if chinese_chars < len(text) * 0.5:
        return False
    # 排除纯标题
    if len(text) < 30 and '。' not in text:
        return False
    return True


def save_jsonl(data, filename):
    """保存为JSONL"""
    path = OUTPUT_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  保存 {len(data)} 条 → {path}")
    return path


# ============================================================
# 1. LCSTS 提取 (新闻正文)
# ============================================================

def extract_from_lcsts(count=500):
    """从LCSTS提取新闻正文作为Human样本"""
    print("\n" + "=" * 60)
    print("从 LCSTS 提取新闻正文")
    print("=" * 60)

    results = []
    lcsts_path = PROJECT_ROOT / 'datasets' / 'external' / 'LCSTS' / 'train.json'

    if not lcsts_path.exists():
        print(f"  文件不存在: {lcsts_path}")
        return results

    all_items = []
    with open(lcsts_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_items.append(json.loads(line))

    print(f"  LCSTS总量: {len(all_items)}")

    # 随机打乱后筛选
    random.shuffle(all_items)

    for item in all_items:
        if len(results) >= count:
            break

        text = clean_text(item.get('input', ''))
        if is_valid_chinese(text, min_len=80, max_len=500):
            results.append({
                'text': text,
                'label': 0,
                'source': 'LCSTS',
                'category': 'news',
                'scenario': 'news_report',
            })

    print(f"  有效提取: {len(results)} 条")
    save_jsonl(results, 'lcsts_news.jsonl')
    return results


# ============================================================
# 2. THUCNews 提取
# ============================================================

def extract_from_thucnews(count=500):
    """从THUCNews提取新闻作为Human样本"""
    print("\n" + "=" * 60)
    print("从 THUCNews 提取新闻")
    print("=" * 60)

    results = []

    # 类别映射
    label_names = {
        0: '体育', 1: '娱乐', 2: '家居', 3: '房产', 4: '教育',
        5: '时尚', 6: '游戏', 7: '科技', 8: '社会', 9: '财经'
    }

    # 优先选择与答辩相关的类别
    priority_labels = {4, 7, 8, 9}  # 教育、科技、社会、财经

    for split in ['train', 'test', 'validation']:
        arrow_path = PROJECT_ROOT / 'datasets' / 'external' / 'THUCNews' / f'thuc_news-{split}.arrow'
        if not arrow_path.exists():
            continue

        try:
            import pyarrow.ipc as ipc
            with open(arrow_path, 'rb') as f:
                reader = ipc.open_stream(f)
                table = reader.read_all()

            texts = table.column('text').to_pylist()
            labels = table.column('label').to_pylist()

            # 按优先级筛选
            priority_items = [(t, l) for t, l in zip(texts, labels) if l in priority_labels]
            other_items = [(t, l) for t, l in zip(texts, labels) if l not in priority_labels]

            random.shuffle(priority_items)
            random.shuffle(other_items)

            candidates = priority_items + other_items

            for text, label in candidates:
                if len(results) >= count:
                    break

                text = clean_text(text)
                if is_valid_chinese(text, min_len=80, max_len=500):
                    cat_name = label_names.get(label, '其他')
                    results.append({
                        'text': text,
                        'label': 0,
                        'source': f'THUCNews-{cat_name}',
                        'category': 'news',
                        'scenario': 'news_report',
                    })

            print(f"  {split}: 提取 {len(results)} 条")

        except Exception as e:
            print(f"  {split}: 加载失败 - {e}")

        if len(results) >= count:
            break

    print(f"  有效提取: {len(results)} 条")
    save_jsonl(results, 'thucnews_news.jsonl')
    return results


# ============================================================
# 3. 新华网在线采集
# ============================================================

def crawl_xinhua(count=200):
    """从新华网采集新闻"""
    print("\n" + "=" * 60)
    print("新华网在线采集")
    print("=" * 60)

    results = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    })

    from bs4 import BeautifulSoup

    channels = [
        ("http://www.news.cn/politics/", "时政"),
        ("http://www.news.cn/fortune/", "财经"),
        ("http://www.news.cn/tech/", "科技"),
        ("http://www.news.cn/local/", "地方"),
        ("http://www.news.cn/edu/", "教育"),
    ]

    for base_url, channel in channels:
        print(f"  频道: {channel}")
        try:
            resp = session.get(base_url, timeout=15)
            if resp.status_code != 200:
                print(f"    请求失败: {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/202' in href and href.endswith('.htm'):
                    full_url = urljoin(base_url, href)
                    if full_url not in links:
                        links.append(full_url)

            collected = 0
            per_channel = count // len(channels)

            for link in links[:per_channel * 2]:  # 多爬一些，筛选后可能不够
                if collected >= per_channel:
                    break
                try:
                    art_resp = session.get(link, timeout=10)
                    if art_resp.status_code != 200:
                        continue

                    art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                    content_div = (
                        art_soup.find('div', id='detail') or
                        art_soup.find('div', class_='main-aticle') or
                        art_soup.find('div', class_='article')
                    )

                    if content_div:
                        paragraphs = content_div.find_all('p')
                        text = ' '.join([clean_text(p.get_text()) for p in paragraphs])

                        if is_valid_chinese(text, min_len=100, max_len=600):
                            results.append({
                                'text': text,
                                'label': 0,
                                'source': f'新华网-{channel}',
                                'category': 'news',
                                'scenario': 'news_report',
                                'url': link,
                            })
                            collected += 1

                    time.sleep(random.uniform(0.8, 1.5))

                except Exception:
                    continue

            print(f"    采集 {collected} 条")
            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print(f"    失败: {e}")

    print(f"  新华网总计: {len(results)} 条")
    if results:
        save_jsonl(results, 'xinhua_news.jsonl')
    return results


# ============================================================
# 4. 模板生成 (通知 + 日常 + 邮件 + 评价)
# ============================================================

def generate_templates(count=400):
    """生成模板数据覆盖各薄弱场景"""
    print("\n" + "=" * 60)
    print("模板生成 (通知/日常/邮件/评价)")
    print("=" * 60)

    results = []

    # === 通知类 ===
    notice_templates = [
        "各学院、各部门：{body}特此通知。",
        "全校师生：{body}请周知。",
        "各位同学：{body}如有疑问请联系相关部门。",
        "全体教职工：{body}望各单位遵照执行。",
    ]
    notice_bodies = [
        "根据学校安排，本学期期末考试将于第18周进行。请各学院提前做好考试安排，通知学生按时参加考试。监考教师应提前15分钟到达考场。",
        "经学校研究决定，自本学期起图书馆开放时间调整为早8点至晚10点。自习室周末正常开放。请合理安排学习时间。",
        "本学期第二轮选课将于下周一开始，请同学们登录教务系统查看剩余名额。选课期间如遇系统问题，可到教务处窗口现场办理。",
        "为做好校园消防安全工作，保卫处将于本月进行消防设施专项检查。请各单位提前做好自查，确保消防通道畅通、灭火器材完好。",
        "本周四下午两点在行政楼三楼会议室召开全体辅导员工作会议，请各学院辅导员准时参加，不得缺席。",
        "2026届本科毕业论文中期检查工作即将开始，请各学院通知指导教师和学生做好准备，按时提交中期检查报告。",
        "关于寒假留校学生登记的通知：需要寒假留校的同学请于本周五前到各学院学工办登记，并填写留校申请表。",
        "为丰富教职工文体生活，工会拟于本月末举办教职工趣味运动会，请各单位积极组队参加，报名截止本周三。",
        "信息中心定于本周六凌晨进行校园网维护升级，届时网络将暂停服务约4小时。请师生提前做好相关安排。",
        "根据教育部要求，本学期将开展本科教学质量审核评估自评工作。请各教学单位认真准备材料，按时完成各项任务。",
    ]

    # 生成所有组合
    notice_combos = [(t, b) for t in notice_templates for b in notice_bodies]
    random.shuffle(notice_combos)
    for tpl, body in notice_combos[:count // 4]:
        results.append({
            'text': tpl.format(body=body),
            'label': 0,
            'source': '模板-通知',
            'category': 'notice',
            'scenario': 'formal_notice',
        })

    # === 日常分享类 ===
    daily_texts = [
        "今天天气真好，出门遛了一圈，看到路边的花都开了，心情一下子就好了起来。",
        "周末终于可以休息了，在家看了一整天剧，虽然有点浪费时间但是真的太爽了。",
        "刚学会做红烧肉，第一次做居然还挺成功的，给室友尝了都说不错，有成就感。",
        "今天去图书馆复习了一整天，效率还可以，把高数的公式基本都过了一遍。",
        "和同学去吃了学校门口新开的那家火锅，味道不错价格也实惠，推荐给大家。",
        "加班到九点终于把报告写完了，明天交给导师看看，希望不用大改。",
        "今天收到了实习公司的回复，让我下周一去面试，有点紧张但也很期待。",
        "感冒了好几天终于好了，以后一定要注意保暖，身体真的是最重要的。",
        "放假回家第一件事就是吃妈妈做的饭菜，在外面吃了一学期食堂，真的想念家里的味道。",
        "今天打球的时候崴了脚，去校医院看了说没大问题，这几天得好好休息了。",
        "快递到了！等了好几天的耳机终于收到了，试了一下音质挺好的，这钱花得值。",
        "同学聚会拍了好多照片，大家好久没见了都有点变化，但聊起天来还是那么开心。",
        "今天在食堂排队排了二十分钟，中午高峰期人真的太多了，下次得早点来。",
        "考完试了！虽然有几道题不太确定，但总体感觉还行，先不想了出去玩一下。",
        "最近在学Python，跟着网上教程做了个小项目，虽然简单但是很有成就感。",
        "搬到新宿舍了，室友都挺好相处的，希望这学期能愉快地度过。",
        "打卡第30天背单词，虽然有时候想偷懒但还是坚持下来了，继续加油。",
        "今天实验终于跑通了，调了三天的参数，感动到想哭。",
        "论文查重通过了！重复率只有百分之八，这段时间的努力没有白费。",
        "下了一场大雪，校园里好漂亮，和同学堆了个雪人拍了好多照片。",
    ]

    # 不重复选择
    random.shuffle(daily_texts)
    for text in daily_texts[:count // 4]:
        results.append({
            'text': text,
            'label': 0,
            'source': '模板-日常分享',
            'category': 'daily',
            'scenario': 'daily_sharing',
        })

    # === 邮件类 ===
    email_texts = [
        "尊敬的老师您好，我是计算机学院大四的学生，想向您咨询一下毕业设计的选题方向。我对自然语言处理比较感兴趣，不知道您这边是否还有相关课题可以选。期待您的回复，谢谢！",
        "老师好，附件是我的毕业论文初稿，请您抽空审阅。按照您上次的建议，我对实验部分做了补充，增加了对比实验的数据。如有问题我随时修改。",
        "您好，我是XX课程的学生，由于家中有急事需要请假两天，已在系统中提交了请假申请，烦请您审批一下。回来后我会及时补上落下的课程内容。",
        "导师好，关于下周一的组会，我准备汇报一下最近的实验进展，主要是模型训练的结果分析。PPT我会提前发给您过目，请问有什么需要特别准备的吗？",
        "老师您好，我已经完成了开题报告的修改，主要修改了研究方法部分和预期成果的描述。修改后的版本已上传到系统，请您查看。",
        "您好，我是来应聘实习生岗位的张三，简历已发送到公司邮箱。我本科就读于XX大学计算机专业，对贵公司的AI研发岗位非常感兴趣。期待能获得面试机会。",
        "老师好，我想报名参加下周的学术报告会，请问还有名额吗？另外想确认一下具体的时间和地点，谢谢。",
        "您好，请问今年研究生复试的具体安排出来了吗？我看到通知说三月中旬，但具体日期还没有公布。想提前做好准备，谢谢！",
    ]

    random.shuffle(email_texts)
    for text in email_texts[:count // 4]:
        results.append({
            'text': text,
            'label': 0,
            'source': '模板-邮件',
            'category': 'email',
            'scenario': 'formal_email',
        })

    # === 评价/推荐类 ===
    review_texts = [
        "用了两个月了，总体来说还不错。续航能力比预期好，充一次电能用两天。屏幕显示效果也很好，拍照清晰度也够用了。就是有时候会有点发热，不过不影响使用。",
        "这本书真的很不错，内容深入浅出，作为入门教材再合适不过了。每个章节后面都有习题，适合自学。推荐给同样在学这门课的同学。",
        "第一次住这家酒店，位置不错离地铁站很近。房间干净整洁，隔音效果也可以。前台服务态度很好，主动帮忙升级了房型。下次来还会选这里。",
        "味道挺好的，汤底很浓郁，肉片也很新鲜。服务员态度不错，上菜速度也快。就是周末人太多了要排队，建议工作日来。人均八十左右，性价比还可以。",
        "买了给妈妈的生日礼物，她很喜欢。质量做工都不错，颜色也好看。快递很快第二天就到了，包装也很用心，作为礼物送人很合适。",
        "这个APP挺好用的，功能比较全面，界面设计也简洁。最近更新后加了几个新功能，用起来更方便了。就是偶尔会闪退，希望开发者能修复一下。",
        "课程内容挺充实的，老师讲得很清楚，基本上跟着做就能学会。作业量适中，有答疑群可以随时问问题。适合零基础入门，推荐。",
        "新开的那家咖啡馆环境不错，很安静适合自习。咖啡味道也可以，还有免费WiFi。就是位置有点偏，离学校有点远。",
    ]

    random.shuffle(review_texts)
    for text in review_texts[:count // 4]:
        results.append({
            'text': text,
            'label': 0,
            'source': '模板-评价',
            'category': 'review',
            'scenario': 'daily_sharing',
        })

    # 打乱
    random.shuffle(results)

    # 统计
    cat_counts = Counter(d['category'] for d in results)
    print(f"  生成总计: {len(results)} 条")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"    {cat}: {cnt}")

    save_jsonl(results, 'template_generated.jsonl')
    return results


# ============================================================
# 合并
# ============================================================

def merge_all():
    """合并所有提取的数据"""
    print("\n" + "=" * 60)
    print("合并所有提取数据")
    print("=" * 60)

    all_data = []
    seen = set()

    for jsonl_file in sorted(OUTPUT_DIR.glob('*.jsonl')):
        if jsonl_file.name == 'all_extracted.jsonl':
            continue
        count = 0
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    key = item['text'][:60]
                    if key not in seen:
                        seen.add(key)
                        all_data.append(item)
                        count += 1
        print(f"  {jsonl_file.name}: {count} 条")

    # 保存
    output_path = OUTPUT_DIR / 'all_extracted.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n合并完成: {len(all_data)} 条 (去重后)")
    print(f"保存至: {output_path}")

    # 同时生成CSV
    import csv
    csv_path = OUTPUT_DIR / 'all_extracted.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'source', 'category', 'scenario'])
        writer.writeheader()
        for item in all_data:
            writer.writerow({
                'text': item['text'],
                'label': item['label'],
                'source': item.get('source', ''),
                'category': item.get('category', ''),
                'scenario': item.get('scenario', ''),
            })
    print(f"CSV格式: {csv_path}")

    # 统计
    cat_counts = Counter(d.get('category', 'unknown') for d in all_data)
    print("\n按类别:")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")

    scenario_counts = Counter(d.get('scenario', 'unknown') for d in all_data)
    print("\n按场景:")
    for sc, cnt in sorted(scenario_counts.items()):
        print(f"  {sc}: {cnt}")

    return all_data


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='提取答辩场景数据')
    parser.add_argument('--source', choices=['lcsts', 'thucnews', 'xinhua', 'templates'],
                       help='指定数据源')
    parser.add_argument('--all', action='store_true', help='执行全部提取')
    parser.add_argument('--merge', action='store_true', help='合并数据')
    parser.add_argument('--count', type=int, default=500, help='每源目标数量')
    args = parser.parse_args()

    print("=" * 60)
    print("答辩场景数据提取")
    print("=" * 60)

    if args.all or args.source == 'lcsts':
        extract_from_lcsts(args.count)

    if args.all or args.source == 'thucnews':
        extract_from_thucnews(args.count)

    if args.all or args.source == 'xinhua':
        crawl_xinhua(min(args.count, 200))

    if args.all or args.source == 'templates':
        generate_templates(args.count)

    if args.all or args.merge:
        merge_all()

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
