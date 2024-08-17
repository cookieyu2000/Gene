import os
import logging
from collections import defaultdict

# 配置日志记录
logging.basicConfig(level=logging.INFO, filename='data_filter.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

input_file = 'data/PubMed_ner_data.txt'  # 原始数据文件路径
output_file = 'data/filtered_data.txt'  # 过滤后的数据文件路径

def is_valid_group(group):
    """
    检查每组数据是否包含所需的三种格式。
    """
    has_t = False
    has_a = False
    has_entities = False

    for line in group:
        parts = line.strip().split('|')
        if len(parts) == 3:
            if parts[1].strip() == 't':
                has_t = True
            elif parts[1].strip() == 'a':
                has_a = True
        elif len(parts) == 5:
            has_entities = True

    return has_t and has_a and has_entities

def filter_data(input_file, output_file):
    groups = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            parts = line.strip().split('|')
            if len(parts) == 3 or len(parts) == 5:
                pmid = parts[0].strip()
                groups[pmid].append(line)
            else:
                logging.warning(f"Skipping line due to unexpected format: {line.strip()}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        skipped_groups_count = 0
        total_groups_count = 0

        for pmid, group in groups.items():
            total_groups_count += 1
            if is_valid_group(group):
                for line in group:
                    outfile.write(line)
                outfile.write('\n')
            else:
                skipped_groups_count += 1
                logging.warning(f"Skipping group for PMID {pmid} due to incomplete data")

        print(f"Total groups processed: {total_groups_count}")
        print(f"Total groups skipped: {skipped_groups_count}")
        print(f"Filtered data saved to: {output_file}")

if __name__ == "__main__":
    filter_data(input_file, output_file)
