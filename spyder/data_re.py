import re
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def split_into_sentences(paragraph):
    """使用正则表达式将段落分割成单独的句子。"""
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(paragraph)
    return sentences

with open("data/sentence_DM_not.txt", 'r', encoding='utf-8') as file:
    data = file.readlines()

articles = {}
labels = {}

def data_processing(data):
    for line in data:
        parts = line.strip().split('|')
        if len(parts) == 3:  # 處理文章内容或者标籤
            pmid, section, content = parts
            pmid = pmid.strip()
            section = section.strip()
            content = content.strip()

            # 根據 section 是否為 'l' 來區分文章內容與標籤
            if section == 'l':  # 處理標籤
                if pmid not in labels:
                    labels[pmid] = {'label': content, 'annotations': []}
                else:
                    labels[pmid]['label'] = content
            else:  # 處理文章內容
                if pmid not in articles:
                    articles[pmid] = {'t': '', 'a': ''}
                articles[pmid][section] = content

        elif len(parts) == 5:  # 處理其他標籤格式
            pmid, start, end, name, biotype = parts
            pmid = pmid.strip()
            start = int(start.strip())
            end = int(end.strip())
            name = name.strip()
            biotype = biotype.strip()
            if pmid not in labels:
                labels[pmid] = {'label': '', 'annotations': []}
            labels[pmid]['annotations'].append({
                'start': start,
                'end': end,
                'name': name,
                'biotype': biotype
            })
    return articles, labels

articles, labels = data_processing(data)

# 初始化保存結果的列表
extracted_data = []
seen_sentences = set()  # 用於跟踪已保存的句子

# 遍歷 labels 以查找符合條件的句子，並使用 tqdm 顯示進度條
for pmid, pmid_data in tqdm(labels.items(), desc="Processing labels"):
    pmid_labels = pmid_data['annotations']
    label_value = pmid_data['label']  # 获取与PMID相关联的label值
    
    for label in pmid_labels:
        # 先檢查 biotype 是否存在
        if 'biotype' in label and label['biotype'] == 'variant':
            paragraph = articles.get(pmid, {}).get('t', '') + " " + articles.get(pmid, {}).get('a', '')
            sentences = split_into_sentences(paragraph)
            
            for sentence in sentences:
                # 找到当前句子中所有的 variant
                variants_in_sentence = list(set([l['name'] for l in pmid_labels if l.get('biotype') == 'variant' and l['name'] in sentence]))
                
                if variants_in_sentence and sentence not in seen_sentences:
                    # 查找同一篇文章中的 gene 和 disease
                    genes = list(set([l['name'] for l in pmid_labels if l.get('biotype') == 'gene' and sentence.find(l['name']) != -1]))
                    diseases = list(set([l['name'] for l in pmid_labels if l.get('biotype') == 'disease' and sentence.find(l['name']) != -1]))

                    # 保存符合條件的句子並加入 seen_sentences 集合
                    if genes and diseases:
                        extracted_data.append([pmid, sentence, ", ".join(variants_in_sentence), ", ".join(genes), ", ".join(diseases), label_value])
                    elif genes:
                        extracted_data.append([pmid, sentence, ", ".join(variants_in_sentence), ", ".join(genes), '', label_value])
                    elif diseases:
                        extracted_data.append([pmid, sentence, ", ".join(variants_in_sentence), '', ", ".join(diseases), label_value])
                    
                    # 將已保存的句子加入到 seen_sentences 集合中，避免重複
                    seen_sentences.add(sentence)

# 將結果寫入 CSV 文件
with open('data/extracted_sentences.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['PMID', 'Sentence', 'Variant', 'Gene', 'Disease', 'Label'])
    csvwriter.writerows(extracted_data)

print("Data extraction complete. Saved to extracted_sentences.csv")
