#!/usr/bin/env python3
"""
构建多样化Human数据集 - 增强版

大幅扩展模板变量，增加随机组合以减少重复
"""
import json
import random
import os
from pathlib import Path
from itertools import product


def load_extracted_human():
    """加载已提取但未使用的Human数据"""
    path = 'datasets/external/extracted_all_for_training.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for item in data['human']:
        source = item.get('source', '')
        # 排除已使用的来源
        if source in ['HC3-Chinese', 'LCSTS']:
            continue
        result.append({
            'text': item['text'],
            'label': 0,
            'source': f"external_{source.lower().replace('-', '_')}",
            'category': 'Human',
            'type': item.get('source_type', 'qa')
        })

    print(f"从外部数据提取: {len(result)} 条")
    return result


# ============ 正式通知 - 扩展版 ============
def generate_notices_v2(count=2000):
    """生成正式通知样本 - 高度多样化"""
    templates = []

    # 会议通知模板集
    times = ['明天上午9点', '本周五下午3点', '下周一早上8:30', '今天下午2点',
             '周三上午10点', '25号下午4点', '后天上午10点', '本月底',
             '下周二下午', '周四晚7点', '本周日上午']
    places = ['会议室', '大会议室', '三楼报告厅', '食堂二楼', 'A座5楼会议室',
              'B栋6楼', '行政楼会议中心', '图书馆报告厅', '体育馆',
              '多媒体教室', '线上腾讯会议', '钉钉群', '公司大堂']
    meetings = ['全体会议', '部门例会', '工作汇报会', '年终总结会',
                '培训会议', '动员大会', '座谈会', '述职会议', '安全会议',
                '业务研讨会', '项目启动会', '月度总结会']
    audiences = ['全体员工', '各位同事', '全体师生', '各部门负责人',
                 '相关人员', '班组长以上干部', '全体党员', '新入职员工']

    for t in times:
        for p in places[:8]:
            for m in meetings[:8]:
                for a in audiences[:5]:
                    templates.append(f"通知：{t}在{p}召开{m}，请{a}准时参加。")

    # 物业/社区通知
    dates = ['即日起', '本月15日', '下周起', '元旦后', '春节前', '明天起',
             '本周六', '下月1号起', '国庆后', '本周末']
    contents = ['电梯将进行年检维护', '小区将进行消杀作业', '水管维修改造',
                '道路施工封闭', '绿化带修剪', '停车场划线施工',
                '外墙粉刷', '管道检修', '消防演练', '垃圾分类培训',
                '电费调价', '物业费催缴', '车位重新分配']
    impacts = ['电梯停运', '热水供应中断', '部分区域封闭', '噪音干扰',
               '通行不便', '临时停水', '临时停电', '网络中断']

    for d in dates:
        for c in contents:
            for i in impacts[:5]:
                templates.append(f"尊敬的业主：{d}{c}，届时{i}，请提前做好准备。")

    # 学校/机构通知
    school_contents = ['期末考试安排已公布', '暑假放假时间确定', '新学期注册缴费开始',
                       '运动会报名开始', '奖学金评定开始', '社团招新进行中',
                       '图书馆闭馆整理', '食堂菜单更新', '校园卡充值点变更']

    for c in school_contents:
        templates.append(f"通知：{c}，请同学们知悉。")
        templates.append(f"各位同学注意：{c}，详情见教务处公告。")

    # 紧急通知
    reasons = ['设备检修', '暴雨预警', '系统维护', '疫情防控需要', '突发停电',
               '管道爆裂', '安全隐患排查', '领导检查', '重要接待']
    actions = ['今日下午停止办公', '活动改期举行', '食堂暂停营业', '班车停运',
               '请假外出', '居家办公', '暂停对外服务', '临时放假']

    for r in reasons:
        for a in actions:
            templates.append(f"紧急通知：因{r}，{a}，请大家注意。")

    # 随机选取并添加变体
    results = []
    random.shuffle(templates)
    for i, text in enumerate(templates[:count]):
        # 添加真实性变体
        if random.random() < 0.3:
            text = text.rstrip('。')
        if random.random() < 0.2:
            text = text.replace('。', '！', 1)
        if random.random() < 0.1:
            text = f"【{random.randint(1,28)}号】" + text
        if random.random() < 0.15:
            text = text.replace('请', '请大家')
        if random.random() < 0.1:
            text = text + " 谢谢配合。"

        results.append({
            'text': text,
            'label': 0,
            'source': 'template_notice',
            'category': 'Human',
            'type': 'formal_notice'
        })

    return results


# ============ 日常社交 - 扩展版 ============
def generate_social_v2(count=3000):
    """生成日常社交样本 - 高度多样化"""
    templates = []

    # 日常分享
    time_words = ['今天', '刚刚', '昨天', '周末', '下班后', '中午', '早上',
                  '刚才', '这两天', '前几天', '上周']
    did_whats = ['去看了电影', '吃了顿火锅', '买了新衣服', '收到快递',
                 '加了个班', '健身了一小时', '追完了剧', '做了个蛋糕',
                 '和朋友逛街', '去医院了', '去了趟超市', '修好了电脑',
                 '换了新手机', '剪了头发', '去公园散步', '学了道新菜',
                 '看了场比赛', '参加了聚会', '去听了演唱会', '拿到驾照了',
                 '搬家了', '去面试', '交房租了', '发工资了']
    feelings = ['太开心了', '好累啊', '还不错', '感觉一般般', '有点失望',
                '超满意', '饿死了', '困得不行', '心情好', '有点烦',
                '太爽了', '好郁闷', '不想动', '超充实', '无聊死了',
                '感动哭了', '气死我了', '太难了', '笑死', '绝望']

    for t in time_words:
        for d in did_whats:
            for f in feelings[:10]:
                templates.append(f"{t}{d}，{f}")

    # 吐槽
    complaints = ['怎么又堵车', '快递还没到', '工资还没发', '又要加班',
                  '手机没电了', '又胖了', '公交怎么这么久不来',
                  '外卖又送错了', 'WiFi又断了', '下雨没带伞',
                  '今天又迟到了', '钱包找不到了', '忘带钥匙了',
                  '电脑又卡了', '又失眠了', '空调坏了', '手机摔了']
    reactions = ['无语了', '要疯了', '不想说话', '想哭', '好烦', '服了',
                 '心态崩了', '直接社死', '想骂人', '不是哥们', '救命']

    for c in complaints:
        for r in reactions[:7]:
            templates.append(f"{c}，{r}")
            templates.append(f"救命，{c}...")
            templates.append(f"不是，{c}？{r}")

    # 记录打卡
    places = ['图书馆', '健身房', '公司', '新餐厅', '咖啡店', '商场',
              '电影院', '博物馆', '美术馆', '网红店', '密室', '游乐园']
    comments = ['绝了', '一般', '还可以', '真的棒', '不太行', '太绝了',
                '推荐', '不推荐', '下次还来', '不会再去了', '巨好吃',
                '一言难尽', '性价比高', '有点贵', '值回票价']

    for p in places:
        for c in comments[:8]:
            templates.append(f"打卡{p}，{c}")

    # 短句
    short_sentences = [
        '好困', '饿了', '下班了', '到家', '晚安', '早安',
        '冲冲冲', '加油', '好耶', '累死了', 'emo了', '无聊',
        '等人中', '在路上', '快到了', '出发', '睡不着',
        '天呐', '我的天', '吃饱了', '又饿了', '好热', '好冷',
        '想吃火锅', '想喝奶茶', '该减肥了', '不想上班', '周末快来',
        '终于周五了', '又是周一', '打工人打工魂', '摸鱼中',
        '偷偷摸鱼', '被发现了', '没事了', '搞定了', '收工',
    ]
    for s in short_sentences:
        templates.append(s)
        if random.random() < 0.3:
            templates.append(s + '～')
        if random.random() < 0.2:
            templates.append(s + '！')

    # 提问
    questions = ['附近有好吃的', '这个怎么弄', '哪里能买到', '有推荐的吗',
                 '什么时候发货', '有人一起吗', '坐标在哪', '这个值得买吗']
    for q in questions:
        templates.append(f"有人知道{q}吗？")
        templates.append(f"{q}，求推荐！")

    random.shuffle(templates)
    results = []
    for text in templates[:count]:
        # 添加真实性变体
        if random.random() < 0.4:
            text = text.rstrip('。！')
        if random.random() < 0.2 and not text.endswith('～'):
            text = text + '～'
        if random.random() < 0.15:
            text = text.replace('！', '！！')
        if random.random() < 0.1:
            text = text.replace('，', random.choice(['，唉，', '，哎，', '，呃，']), 1)
        if random.random() < 0.05:
            # 添加哈哈
            text = text + random.choice(['哈哈', '哈哈哈', '笑死'])

        results.append({
            'text': text,
            'label': 0,
            'source': 'template_social',
            'category': 'Human',
            'type': 'daily_social'
        })

    return results


# ============ 电商评价 - 扩展版 ============
def generate_reviews_v2(count=2000):
    """生成电商评价样本 - 高度多样化"""
    templates = []

    # 好评要素
    qualities = ['质量很好', '材质不错', '做工精细', '穿着舒服', '手感很好',
                 '颜色正', '尺码标准', '很实用', '比想象中好', '物超所值',
                 '款式好看', '面料舒适', '版型正', '很厚实', '透气性好',
                 '不起球', '不掉色', '弹性好', '很软', '质感不错',
                 '细节处理好', '做工细腻', '用料足', '有分量', '很精致']
    logistics = ['物流很快', '发货快', '包装完好', '第二天就到了', '快递给力',
                 '物流超快', '隔天到', '包装严实', '没有破损', '送货上门']
    services = ['客服态度好', '有问必答', '售后不错', '很耐心', '服务周到',
                '回复及时', '解决问题快', '客服专业', '态度很好']
    recommendations = ['推荐购买', '会回购', '已推荐给朋友', '下次还来',
                       '好评', '五星好评', '非常满意', '值得购买', '闭眼入']

    for q in qualities[:15]:
        for l in logistics[:6]:
            templates.append(f"{q}，{l}")
        for r in recommendations[:5]:
            templates.append(f"{q}，{r}")

    # 回购评价
    for n in ['二', '三', '四', '五', 'N', '无数']:
        for q in qualities[:10]:
            templates.append(f"第{n}次买了，{q}")
            templates.append(f"回购{n}次了，一如既往的好")

    # 中评
    neutrals = ['和图片有点色差', '比预期小一点', '味道一般', '还需要观察',
                '中规中矩', '没有惊喜', '凑合用', '比想象中薄', '有点味道',
                '线头有点多', '尺码偏小', '尺码偏大', '和描述有差距',
                '性价比还行', '就那样吧', '一般般']
    for n in neutrals:
        templates.append(f"还可以吧，{n}")
        templates.append(f"一般般，{n}")
        templates.append(f"{n}，不过价格便宜凑合用")

    # 差评
    complaints = ['质量太差', '和图片不符', '发错货了', '有破损', '等了好久',
                  '尺码不对', '掉色严重', '味道很大', '做工粗糙', '线头多',
                  '有瑕疵', '材质不好', '太薄了', '不值这个价']
    results_neg = ['申请退货', '不会再买了', '太失望了', '浪费钱', '差评',
                   '退了', '不推荐', '慎买', '避雷']
    for c in complaints[:10]:
        for r in results_neg[:5]:
            templates.append(f"{c}，{r}")

    # 追评
    times = ['一周', '半个月', '一个月', '几天', '两个月', '用了一段时间']
    updates = ['还是不错的', '质量OK', '用着还行', '没什么问题', '挺满意',
               '已经坏了', '掉色了', '起球了', '变形了', '还好']
    for t in times:
        for u in updates[:6]:
            templates.append(f"用了{t}，{u}")
            templates.append(f"追评：{u}")

    random.shuffle(templates)
    results = []
    for text in templates[:count]:
        if random.random() < 0.3:
            text = text.rstrip('。')
        if random.random() < 0.1:
            # 打错字
            typos = [('的', '得'), ('得', '的'), ('吧', '把')]
            for old, new in typos:
                if old in text and random.random() < 0.3:
                    text = text.replace(old, new, 1)
                    break
        if random.random() < 0.15:
            text = text.replace('很', '很很')
        if random.random() < 0.1:
            text = text + " " + random.choice(['👍', '⭐⭐⭐⭐⭐', ''])

        results.append({
            'text': text,
            'label': 0,
            'source': 'template_review',
            'category': 'Human',
            'type': 'ecommerce_review'
        })

    return results


# ============ 温馨提示 - 扩展版 ============
def generate_tips_v2(count=1000):
    """生成温馨提示样本 - 高度多样化"""
    tips = [
        '天气转凉注意保暖', '出门记得带伞', '疫情期间请佩戴口罩',
        '注意个人财物安全', '保持安全距离', '请勿大声喧哗',
        '禁止吸烟', '请排队等候', '保持环境整洁', '注意脚下安全',
        '小心地滑', '老人小孩注意安全', '请勿触摸展品',
        '营业时间9点到6点', '节假日照常营业', '停车请锁好车门',
        '请自觉投币', '垃圾请分类投放', '节约用水人人有责',
        '文明出行从我做起', '请勿在此停放车辆', '禁止携带宠物入内',
        '请走人行道', '过马路请看红绿灯', '上下楼梯靠右行',
        '电梯满员请等下一班', '请勿奔跑', '轻拿轻放',
        '保持安静', '请勿插队', '先下后上', '注意台阶',
        '小心碰头', '当心玻璃', '禁止拍照', '请勿倚靠',
        '此处有监控', '请爱护公共设施', '请勿乱扔垃圾',
        '洗手间在左边', '出口在右边', '紧急出口在后方',
        '请系好安全带', '请勿携带易燃易爆物品', '请出示健康码',
        '请配合测温', '禁止酒后驾驶', '开车不看手机',
        '高考期间请保持安静', '考试期间禁止喧哗', '请勿带食物入内',
        '饮料请放在外面', '宠物禁止入内', '请勿喂食动物',
    ]

    prefixes = ['温馨提示：', '友情提醒：', '请注意：', '提醒各位：',
                '小贴士：', '注意事项：', '安全提示：', '善意提醒：', '']
    suffixes = ['', '谢谢配合！', '感谢理解。', '请大家遵守。', '谢谢！']

    templates = []
    for tip in tips:
        for prefix in prefixes:
            for suffix in suffixes[:3]:
                templates.append(f"{prefix}{tip}{suffix}")

    random.shuffle(templates)
    results = []
    for text in templates[:count]:
        if random.random() < 0.2:
            text = '【' + text.strip('：') + '】'
        if random.random() < 0.1:
            text = '⚠️' + text

        results.append({
            'text': text,
            'label': 0,
            'source': 'template_tip',
            'category': 'Human',
            'type': 'warm_tip'
        })

    return results


def main():
    print("=" * 60)
    print("构建多样化Human数据集 - 增强版")
    print("=" * 60)

    all_samples = []

    # 1. 从外部数据提取未使用的Human样本
    print("\n1. 提取外部数据...")
    external = load_extracted_human()
    all_samples.extend(external)

    # 2. 生成正式通知
    print("\n2. 生成正式通知样本...")
    notices = generate_notices_v2(2500)  # 多生成一些
    all_samples.extend(notices)
    print(f"   生成: {len(notices)} 条")

    # 3. 生成日常社交
    print("\n3. 生成日常社交样本...")
    social = generate_social_v2(4000)
    all_samples.extend(social)
    print(f"   生成: {len(social)} 条")

    # 4. 生成电商评价
    print("\n4. 生成电商评价样本...")
    reviews = generate_reviews_v2(2500)
    all_samples.extend(reviews)
    print(f"   生成: {len(reviews)} 条")

    # 5. 生成温馨提示
    print("\n5. 生成温馨提示样本...")
    tips = generate_tips_v2(1500)
    all_samples.extend(tips)
    print(f"   生成: {len(tips)} 条")

    # 去重
    print("\n6. 去重...")
    seen = set()
    unique = []
    for s in all_samples:
        text = s['text'].strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(s)
    all_samples = unique
    print(f"   去重后: {len(all_samples)} 条")

    # 统计
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)
    from collections import Counter
    sources = Counter(s['source'] for s in all_samples)
    types = Counter(s['type'] for s in all_samples)

    print("\n按来源:")
    for s, c in sources.most_common():
        print(f"  {s}: {c}")

    print("\n按类型:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")

    # 保存
    output_dir = Path('datasets/human_supplement')
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / 'diverse_human_samples.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\n保存到: {output_path}")
    print(f"总计: {len(all_samples)} 条多样化Human样本")

    # 同时生成CSV格式
    import csv
    csv_path = output_dir / 'diverse_human_samples.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label', 'source', 'category', 'type'])
        writer.writeheader()
        writer.writerows(all_samples)
    print(f"CSV格式: {csv_path}")


if __name__ == '__main__':
    main()
