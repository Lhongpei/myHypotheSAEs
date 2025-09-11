#!/usr/bin/env python3
"""
balance_jsonl.py
从原始 JSONL 文件中抽取 label 分布均匀的子集
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    """返回 list[dict]，逐行读取 jsonl"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(records, path):
    """将 list[dict] 写回 jsonl"""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def balanced_sample(records, seed=42):
    """
    按 label 均匀抽样：每种 label 取 min(label_count) 条
    返回新的 list[dict]
    """
    random.seed(seed)

    # 按 label 分组
    buckets = defaultdict(list)
    for rec in records:
        buckets[rec['label']].append(rec)

    # 每种标签最少出现次数
    k = min(len(v) for v in buckets.values())

    subset = []
    for label, bucket in buckets.items():
        subset.extend(random.sample(bucket, k))

    # 打乱顺序（可选）
    random.shuffle(subset)
    return subset

def main(src_path, dst_path='balanced_subset.jsonl'):
    src_path = Path(src_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    records = load_jsonl(src_path)
    if not records:
        print("文件为空")
        return

    subset = balanced_sample(records)
    save_jsonl(subset, dst_path)

    # 打印分布
    stats = defaultdict(int)
    for r in subset:
        stats[r['label']] += 1
    print("子集标签分布:", dict(stats))
    print(f"已写入 {dst_path}，共 {len(subset)} 条记录")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='抽取 label 均匀分布的 JSONL 子集')
    parser.add_argument('input', help='原始 JSONL 文件')
    parser.add_argument('-o', '--output', default='balanced_subset.jsonl',
                        help='输出文件名（默认 balanced_subset.jsonl）')
    args = parser.parse_args()
    main(args.input, args.output)