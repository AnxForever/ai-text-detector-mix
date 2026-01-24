"""
智能提取短片段脚本
目标：从现有9,170条长AI文本中提取2,000条语义完整的短片段
策略：按段落分割，提取完整段落组合，确保语义完整性
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
import json
from datetime import datetime
from typing import List, Tuple


def split_into_paragraphs(text: str) -> List[str]:
    """
    将文本分割成段落

    策略：
    1. 按双换行符分割
    2. 如果没有双换行符，按单换行符分割
    3. 如果都没有，按句号分割成句子组

    Returns:
        段落列表
    """
    # 先尝试双换行符
    if '\n\n' in text:
        paragraphs = text.split('\n\n')
    elif '\n' in text:
        paragraphs = text.split('\n')
    else:
        # 按句号分割，每3-5句组成一个"段落"
        sentences = text.replace('。', '。\n').split('\n')
        paragraphs = []
        current_para = []
        for sent in sentences:
            current_para.append(sent)
            if len(current_para) >= random.randint(3, 5):
                paragraphs.append(''.join(current_para))
                current_para = []
        if current_para:
            paragraphs.append(''.join(current_para))

    # 清理空段落
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def extract_segment_from_start(paragraphs: List[str], target_min: int = 400, target_max: int = 600) -> str:
    """
    从文本开头提取片段

    策略：累加段落直到长度合适

    Args:
        paragraphs: 段落列表
        target_min: 目标最小长度
        target_max: 目标最大长度

    Returns:
        提取的片段，如果无法提取返回空字符串
    """
    if not paragraphs:
        return ""

    current_length = 0
    selected_paras = []

    for para in paragraphs:
        para_len = len(para)

        # 如果加上这个段落会超出最大长度
        if current_length + para_len > target_max:
            # 如果已经达到最小长度，返回当前结果
            if current_length >= target_min:
                break
            # 如果还没达到最小长度，但加上会超出最大长度
            # 尝试截断这个段落
            elif current_length > 0:
                remaining = target_max - current_length
                if remaining >= 100:  # 至少要有100字符的空间
                    # 截断到合适的句子边界
                    truncated = truncate_to_sentence(para, remaining)
                    if truncated:
                        selected_paras.append(truncated)
                break
            else:
                # 第一个段落就太长，直接截断
                truncated = truncate_to_sentence(para, target_max)
                if truncated:
                    selected_paras.append(truncated)
                break

        selected_paras.append(para)
        current_length += para_len

    segment = '\n\n'.join(selected_paras)

    # 检查长度是否合适
    if target_min <= len(segment) <= target_max:
        return segment
    else:
        return ""


def extract_segment_from_middle(paragraphs: List[str], target_min: int = 400, target_max: int = 600) -> str:
    """
    从文本中间提取片段

    策略：从中间位置开始累加段落

    Returns:
        提取的片段
    """
    if len(paragraphs) < 2:
        return ""

    # 从中间开始
    mid_point = len(paragraphs) // 2
    start_idx = max(1, mid_point - 1)  # 避免从开头或结尾开始

    current_length = 0
    selected_paras = []

    for i in range(start_idx, len(paragraphs)):
        para = paragraphs[i]
        para_len = len(para)

        if current_length + para_len > target_max:
            if current_length >= target_min:
                break
            elif current_length > 0:
                remaining = target_max - current_length
                if remaining >= 100:
                    truncated = truncate_to_sentence(para, remaining)
                    if truncated:
                        selected_paras.append(truncated)
                break
            else:
                truncated = truncate_to_sentence(para, target_max)
                if truncated:
                    selected_paras.append(truncated)
                break

        selected_paras.append(para)
        current_length += para_len

    segment = '\n\n'.join(selected_paras)

    if target_min <= len(segment) <= target_max:
        return segment
    else:
        return ""


def extract_segment_from_end(paragraphs: List[str], target_min: int = 400, target_max: int = 600) -> str:
    """
    从文本结尾提取片段

    策略：从后往前累加段落

    Returns:
        提取的片段
    """
    if not paragraphs:
        return ""

    current_length = 0
    selected_paras = []

    # 从后往前遍历
    for para in reversed(paragraphs):
        para_len = len(para)

        if current_length + para_len > target_max:
            if current_length >= target_min:
                break
            elif current_length > 0:
                remaining = target_max - current_length
                if remaining >= 100:
                    truncated = truncate_to_sentence(para, remaining, from_end=True)
                    if truncated:
                        selected_paras.insert(0, truncated)
                break
            else:
                truncated = truncate_to_sentence(para, target_max, from_end=True)
                if truncated:
                    selected_paras.insert(0, truncated)
                break

        selected_paras.insert(0, para)
        current_length += para_len

    segment = '\n\n'.join(selected_paras)

    if target_min <= len(segment) <= target_max:
        return segment
    else:
        return ""


def truncate_to_sentence(text: str, max_length: int, from_end: bool = False) -> str:
    """
    截断到完整的句子

    Args:
        text: 要截断的文本
        max_length: 最大长度
        from_end: 是否从末尾截断（True）还是从开头截断（False）

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    if from_end:
        # 从末尾截断：保留后面的内容
        truncated = text[-max_length:]
        # 找到第一个句号后的位置
        first_period = truncated.find('。')
        if first_period != -1 and first_period < len(truncated) - 1:
            return truncated[first_period + 1:].strip()
        else:
            return truncated
    else:
        # 从开头截断：保留前面的内容
        truncated = text[:max_length]
        # 找到最后一个句号
        last_period = truncated.rfind('。')
        if last_period != -1:
            return truncated[:last_period + 1]
        else:
            return truncated


def extract_segments_from_text(
    text: str,
    source_info: dict,
    target_min: int = 400,
    target_max: int = 600
) -> List[dict]:
    """
    从一篇长文本中提取多个片段

    策略：
    1. 提取开头片段
    2. 提取中间片段
    3. 提取结尾片段

    Args:
        text: 原始文本
        source_info: 来源信息
        target_min: 目标最小长度
        target_max: 目标最大长度

    Returns:
        片段列表
    """
    # 只处理长文本（>1000字符）
    if len(text) < 1000:
        return []

    paragraphs = split_into_paragraphs(text)

    if len(paragraphs) < 2:
        return []

    segments = []

    # 策略1：从开头提取
    start_segment = extract_segment_from_start(paragraphs, target_min, target_max)
    if start_segment:
        segments.append({
            'text_content': start_segment,
            'length': len(start_segment),
            'source_api': source_info.get('source_api', 'unknown'),
            'original_topic': source_info.get('topic', ''),
            'attribute': source_info.get('attribute', '说明'),
            'extraction_strategy': 'from_start',
            'original_length': len(text),
            'generation_time': datetime.now().isoformat()
        })

    # 策略2：从中间提取（只对长文本）
    if len(text) >= 1500 and len(paragraphs) >= 4:
        middle_segment = extract_segment_from_middle(paragraphs, target_min, target_max)
        if middle_segment and middle_segment != start_segment:
            segments.append({
                'text_content': middle_segment,
                'length': len(middle_segment),
                'source_api': source_info.get('source_api', 'unknown'),
                'original_topic': source_info.get('topic', ''),
                'attribute': source_info.get('attribute', '说明'),
                'extraction_strategy': 'from_middle',
                'original_length': len(text),
                'generation_time': datetime.now().isoformat()
            })

    # 策略3：从结尾提取（只对超长文本）
    if len(text) >= 2000 and len(paragraphs) >= 6:
        end_segment = extract_segment_from_end(paragraphs, target_min, target_max)
        if end_segment and end_segment not in [start_segment, middle_segment if len(segments) > 1 else None]:
            segments.append({
                'text_content': end_segment,
                'length': len(end_segment),
                'source_api': source_info.get('source_api', 'unknown'),
                'original_topic': source_info.get('topic', ''),
                'attribute': source_info.get('attribute', '说明'),
                'extraction_strategy': 'from_end',
                'original_length': len(text),
                'generation_time': datetime.now().isoformat()
            })

    return segments


def extract_segments_from_dataset(
    input_file: str = "datasets/final/parallel_dataset_cleaned.csv",
    target_count: int = 2000,
    target_min: int = 400,
    target_max: int = 600
):
    """
    从数据集中提取短片段

    Args:
        input_file: 输入文件路径
        target_count: 目标片段数
        target_min: 最小长度
        target_max: 最大长度
    """
    print("=" * 70, flush=True)
    print("智能片段提取器 - 从长AI文本中提取短片段", flush=True)
    print("=" * 70, flush=True)

    # 读取数据集
    print(f"\n[1/4] 加载数据集: {input_file}", flush=True)
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"  ✓ 加载完成: {len(df)} 条文本", flush=True)
    print(f"    平均长度: {df['length'].mean():.0f} 字符", flush=True)
    print(f"    长度范围: {df['length'].min()} - {df['length'].max()}", flush=True)

    # 筛选长文本（>1000字符）
    long_texts = df[df['length'] >= 1000].copy()
    print(f"\n[2/4] 筛选长文本 (≥1000字符)", flush=True)
    print(f"  ✓ 找到 {len(long_texts)} 条长文本", flush=True)

    # 提取片段
    print(f"\n[3/4] 提取片段 (目标: {target_count} 条)", flush=True)
    all_segments = []

    # 打乱顺序，确保随机性
    long_texts = long_texts.sample(frac=1, random_state=42).reset_index(drop=True)

    for idx, row in long_texts.iterrows():
        if len(all_segments) >= target_count:
            break

        text = row['text_content']
        source_info = {
            'source_api': row['source_api'],
            'topic': row.get('topic', ''),
            'attribute': row.get('attribute', '说明')
        }

        segments = extract_segments_from_text(text, source_info, target_min, target_max)
        all_segments.extend(segments)

        # 进度显示
        if (idx + 1) % 100 == 0:
            print(f"\r  进度: {idx + 1}/{len(long_texts)} 文本 | "
                  f"已提取: {len(all_segments)}/{target_count} 片段 "
                  f"({len(all_segments)/target_count*100:.1f}%)",
                  end='', flush=True)

    print(f"\n  ✓ 提取完成: {len(all_segments)} 条片段", flush=True)

    # 如果片段数超过目标，随机采样
    if len(all_segments) > target_count:
        print(f"\n片段数({len(all_segments)})超过目标({target_count})，随机采样...", flush=True)
        random.shuffle(all_segments)
        all_segments = all_segments[:target_count]
        print(f"  ✓ 采样完成: {len(all_segments)} 条", flush=True)

    # 保存结果
    print(f"\n[4/4] 保存结果", flush=True)
    df_segments = pd.DataFrame(all_segments)

    # 创建输出目录
    os.makedirs("datasets/short_ai_texts", exist_ok=True)

    # 保存CSV
    output_file = "datasets/short_ai_texts/extracted_segments_2000.csv"
    df_segments.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ 数据已保存: {output_file}", flush=True)

    # 保存元数据
    metadata = {
        'name': '提取的短片段数据集',
        'created_at': datetime.now().isoformat(),
        'total_samples': len(df_segments),
        'source_file': input_file,
        'avg_length': float(df_segments['length'].mean()),
        'min_length': int(df_segments['length'].min()),
        'max_length': int(df_segments['length'].max()),
        'length_std': float(df_segments['length'].std()),
        'extraction_strategies': df_segments['extraction_strategy'].value_counts().to_dict(),
        'api_distribution': df_segments['source_api'].value_counts().to_dict()
    }

    metadata_file = "datasets/short_ai_texts/extracted_segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 元数据已保存: {metadata_file}", flush=True)

    # 统计信息
    print("\n" + "=" * 70, flush=True)
    print("提取统计", flush=True)
    print("=" * 70, flush=True)

    print(f"\n数据集信息:", flush=True)
    print(f"  样本数: {len(df_segments)}", flush=True)
    print(f"  平均长度: {df_segments['length'].mean():.0f} ± {df_segments['length'].std():.0f} 字符", flush=True)
    print(f"  长度范围: {df_segments['length'].min()} - {df_segments['length'].max()}", flush=True)

    print(f"\n提取策略分布:", flush=True)
    for strategy, count in df_segments['extraction_strategy'].value_counts().items():
        print(f"  {strategy}: {count} ({count/len(df_segments)*100:.1f}%)", flush=True)

    print(f"\nAPI来源分布:", flush=True)
    for api, count in df_segments['source_api'].value_counts().items():
        print(f"  {api}: {count} ({count/len(df_segments)*100:.1f}%)", flush=True)

    print(f"\n长度分布:", flush=True)
    bins = [0, 400, 450, 500, 550, 600, 1000]
    hist = pd.cut(df_segments['length'], bins=bins).value_counts().sort_index()
    for interval, count in hist.items():
        print(f"  {interval}: {count} ({count/len(df_segments)*100:.1f}%)", flush=True)

    return df_segments


if __name__ == "__main__":
    print("\n开始从长AI文本中提取短片段...\n", flush=True)

    # 提取2000条片段
    segments = extract_segments_from_dataset(
        input_file="datasets/final/parallel_dataset_cleaned.csv",
        target_count=2000,
        target_min=400,
        target_max=600
    )

    print("\n" + "=" * 70, flush=True)
    print("提取完成！", flush=True)
    print("=" * 70, flush=True)
    print("\n下一步:", flush=True)
    print("  1. 等待 generate_short_ai_texts.py 完成（生成3000条短文本）", flush=True)
    print("  2. 合并两个短文本数据集", flush=True)
    print("  3. 与原始9,170条AI文本合并", flush=True)
    print("  4. 重新划分train/val/test", flush=True)
