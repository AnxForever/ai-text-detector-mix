"""
增强的Markdown格式处理函数库

功能：
1. 全面的markdown格式检测和去除
2. 详细的格式特征分析
3. 与现有 remove_format_bias.py 兼容

作者：Format Debiasing Team
日期：2026-01-11
"""

import re
from typing import Dict, List, Union


def remove_markdown_comprehensive(text: str) -> str:
    """
    更全面的markdown去除，包括所有常见格式

    支持的格式：
    - 标题（# ## ###）
    - 加粗（**text**）
    - 斜体（_text_ 或 *text*）
    - 列表（- * 1.）
    - 分割线（--- *** ___）
    - 代码块（```code```）
    - 内联代码（`code`）
    - 表格（| col | col |）
    - 引用（> quote）
    - 图片（![alt](url)）
    - 链接（[text](url)）

    Args:
        text: 输入文本

    Returns:
        去除markdown格式后的纯文本
    """
    if not isinstance(text, str):
        return text

    # 定义所有替换模式（顺序很重要！）
    patterns = [
        # 1. 代码块（需要最先处理，避免内部内容被误匹配）
        (r'```[^`]*```', ''),  # 多行代码块
        (r'`([^`]+)`', r'\1'),  # 内联代码（保留内容）

        # 2. 图片和链接
        (r'!\[[^\]]*\]\([^\)]+\)', ''),  # 图片（完全删除）
        (r'\[([^\]]+)\]\([^\)]+\)', r'\1'),  # 链接（保留文字部分）

        # 3. 标题
        (r'^#{1,6}\s+', ''),  # 标题符号

        # 4. 加粗和斜体
        (r'\*\*\*([^*]+)\*\*\*', r'\1'),  # 加粗+斜体
        (r'\*\*([^*]+)\*\*', r'\1'),  # 加粗
        (r'__([^_]+)__', r'\1'),  # 加粗（下划线）
        (r'_([^_]+)_', r'\1'),  # 斜体
        (r'\*([^*]+)\*', r'\1'),  # 斜体（星号）

        # 5. 列表
        (r'^[\-\*\+]\s+', ''),  # 无序列表
        (r'^\d+\.\s+', ''),  # 有序列表

        # 6. 引用
        (r'^>\s+', ''),  # 引用符号

        # 7. 分割线
        (r'^[\-*_]{3,}$', ''),  # 分割线

        # 8. 表格
        (r'^\|[^\n]+\|$', ''),  # 表格行
        (r'^\|[\s\-:]+\|$', ''),  # 表格分隔符

        # 9. 删除线和高亮
        (r'~~([^~]+)~~', r'\1'),  # 删除线
        (r'==([^=]+)==', r'\1'),  # 高亮

        # 10. HTML标签（简单处理）
        (r'<[^>]+>', ''),  # HTML标签

        # 11. 清理多余空行和空格
        (r'\n{3,}', '\n\n'),  # 多余空行
        (r'[ \t]+\n', '\n'),  # 行尾空格
    ]

    # 依次应用所有模式
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

    # 最后清理首尾空白
    return text.strip()


def has_markdown(text: str) -> bool:
    """
    基础markdown格式检测（与现有代码兼容）

    检测常见的markdown格式标记

    Args:
        text: 输入文本

    Returns:
        是否包含markdown格式
    """
    if not isinstance(text, str):
        return False

    # 基础检测模式（与原始 remove_format_bias.py 保持一致）
    patterns = [
        r'\*\*[^*]+\*\*',  # 加粗
        r'^#{1,6}\s',       # 标题
        r'^[\-\*\+]\s',     # 列表
        r'^[\-*_]{3,}$',    # 分割线
        r'```',             # 代码块
    ]

    return any(re.search(p, str(text), re.MULTILINE) for p in patterns)


def has_markdown_detailed(text: str) -> Dict[str, Union[bool, List[str], int, float]]:
    """
    详细的markdown特征分析

    返回文本中包含的所有markdown特征类型和统计信息

    Args:
        text: 输入文本

    Returns:
        字典包含：
        - has_any: 是否包含任何markdown格式
        - types: 包含的格式类型列表
        - count: 格式类型数量
        - diversity: 格式多样性（0-1）
        - details: 每种格式的详细统计
    """
    if not isinstance(text, str):
        return {
            'has_any': False,
            'types': [],
            'count': 0,
            'diversity': 0.0,
            'details': {}
        }

    # 检测各种格式特征
    features = {
        'title': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE)),
        'bold': bool(re.search(r'\*\*[^*]+\*\*|__[^_]+__', text)),
        'italic': bool(re.search(r'_[^_]+_|\*[^*]+\*', text)),
        'list': bool(re.search(r'^[\-\*\+]\s|^\d+\.\s', text, re.MULTILINE)),
        'code_block': bool(re.search(r'```', text)),
        'inline_code': bool(re.search(r'`[^`]+`', text)),
        'divider': bool(re.search(r'^[\-*_]{3,}$', text, re.MULTILINE)),
        'quote': bool(re.search(r'^>\s', text, re.MULTILINE)),
        'link': bool(re.search(r'\[[^\]]+\]\([^\)]+\)', text)),
        'image': bool(re.search(r'!\[[^\]]*\]\([^\)]+\)', text)),
        'table': bool(re.search(r'^\|[^\n]+\|', text, re.MULTILINE)),
    }

    # 统计详细信息
    details = {}

    # 标题统计
    if features['title']:
        titles = re.findall(r'^(#{1,6})\s', text, re.MULTILINE)
        details['title'] = {
            'count': len(titles),
            'levels': list(set(len(t) for t in titles))
        }

    # 加粗统计
    if features['bold']:
        bolds = re.findall(r'\*\*[^*]+\*\*|__[^_]+__', text)
        details['bold'] = {'count': len(bolds)}

    # 列表统计
    if features['list']:
        lists = re.findall(r'^([\-\*\+]|\d+\.)\s', text, re.MULTILINE)
        details['list'] = {'count': len(lists)}

    # 代码块统计
    if features['code_block']:
        code_blocks = re.findall(r'```[^`]*```', text)
        details['code_block'] = {'count': len(code_blocks)}

    # 内联代码统计
    if features['inline_code']:
        inline_codes = re.findall(r'`[^`]+`', text)
        details['inline_code'] = {'count': len(inline_codes)}

    # 链接统计
    if features['link']:
        links = re.findall(r'\[[^\]]+\]\([^\)]+\)', text)
        details['link'] = {'count': len(links)}

    # 图片统计
    if features['image']:
        images = re.findall(r'!\[[^\]]*\]\([^\)]+\)', text)
        details['image'] = {'count': len(images)}

    # 提取有效的格式类型
    active_types = [k for k, v in features.items() if v]

    return {
        'has_any': any(features.values()),
        'types': active_types,
        'count': len(active_types),
        'diversity': len(active_types) / len(features),  # 格式多样性得分
        'details': details
    }


def compare_format_features(text1: str, text2: str) -> Dict[str, any]:
    """
    对比两段文本的格式特征差异

    用于验证格式去除的效果

    Args:
        text1: 第一段文本（通常是原始文本）
        text2: 第二段文本（通常是处理后文本）

    Returns:
        格式特征对比结果
    """
    features1 = has_markdown_detailed(text1)
    features2 = has_markdown_detailed(text2)

    # 计算差异
    removed_types = set(features1['types']) - set(features2['types'])
    added_types = set(features2['types']) - set(features1['types'])
    common_types = set(features1['types']) & set(features2['types'])

    return {
        'original': features1,
        'processed': features2,
        'removed_types': list(removed_types),
        'added_types': list(added_types),
        'common_types': list(common_types),
        'format_reduction': features1['count'] - features2['count'],
        'diversity_change': features2['diversity'] - features1['diversity']
    }


def batch_remove_markdown(texts: List[str], show_progress: bool = False) -> List[str]:
    """
    批量去除markdown格式

    Args:
        texts: 文本列表
        show_progress: 是否显示进度

    Returns:
        去除格式后的文本列表
    """
    results = []
    total = len(texts)

    for i, text in enumerate(texts):
        cleaned = remove_markdown_comprehensive(text)
        results.append(cleaned)

        if show_progress and (i + 1) % 1000 == 0:
            print(f"  已处理: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)", flush=True)

    return results


def get_format_statistics(texts: List[str]) -> Dict[str, any]:
    """
    获取文本集合的格式统计信息

    Args:
        texts: 文本列表

    Returns:
        格式统计信息
    """
    total_count = len(texts)
    has_markdown_count = sum(1 for text in texts if has_markdown(text))

    # 统计各种格式类型的出现频率
    format_type_counts = {
        'title': 0,
        'bold': 0,
        'italic': 0,
        'list': 0,
        'code_block': 0,
        'inline_code': 0,
        'divider': 0,
        'quote': 0,
        'link': 0,
        'image': 0,
        'table': 0,
    }

    for text in texts:
        details = has_markdown_detailed(text)
        for format_type in details['types']:
            if format_type in format_type_counts:
                format_type_counts[format_type] += 1

    # 计算百分比
    format_type_percentages = {
        k: v / total_count * 100 for k, v in format_type_counts.items()
    }

    return {
        'total_count': total_count,
        'has_markdown_count': has_markdown_count,
        'markdown_rate': has_markdown_count / total_count if total_count > 0 else 0.0,
        'format_type_counts': format_type_counts,
        'format_type_percentages': format_type_percentages
    }


# 测试函数
def test_format_handler():
    """测试格式处理函数"""

    # 测试文本（包含多种格式）
    test_text = """## 人工智能简介

**人工智能**（AI）是计算机科学的重要分支。

### 核心技术：
- 机器学习
- 深度学习
- *自然语言处理*

---

这里是[链接](https://example.com)和![图片](image.jpg)。

```python
def hello():
    print("Hello AI")
```

内联代码：`import torch`

> 这是引用内容

| 算法 | 准确率 |
|------|--------|
| BERT | 99.5%  |
"""

    print("="*60)
    print("格式处理函数测试")
    print("="*60)

    # 测试1：基础检测
    print("\n[测试1] 基础markdown检测")
    has_md = has_markdown(test_text)
    print(f"是否包含markdown: {has_md}")

    # 测试2：详细分析
    print("\n[测试2] 详细格式分析")
    details = has_markdown_detailed(test_text)
    print(f"包含的格式类型: {details['types']}")
    print(f"格式类型数量: {details['count']}")
    print(f"格式多样性: {details['diversity']:.2%}")
    print(f"\n详细统计:")
    for fmt_type, info in details['details'].items():
        print(f"  {fmt_type}: {info}")

    # 测试3：格式去除
    print("\n[测试3] 格式去除")
    cleaned = remove_markdown_comprehensive(test_text)
    print(f"原始文本长度: {len(test_text)}")
    print(f"清理后长度: {len(cleaned)}")
    print(f"\n清理后文本预览:\n{cleaned[:200]}...")

    # 测试4：对比分析
    print("\n[测试4] 格式对比分析")
    comparison = compare_format_features(test_text, cleaned)
    print(f"去除的格式类型: {comparison['removed_types']}")
    print(f"格式减少数量: {comparison['format_reduction']}")

    print("\n✓ 所有测试完成！")


if __name__ == "__main__":
    test_format_handler()
