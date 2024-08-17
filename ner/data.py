import pandas as pd
from transformers import AutoTokenizer
import torch
import warnings
from tqdm import tqdm  # 引入 tqdm 用於顯示進度條
from label import label_map  # 從 label.py 引入 label_map
from concurrent.futures import ThreadPoolExecutor, as_completed
import random  # 用於隨機選取一筆資料

warnings.filterwarnings('ignore')

# 讀取資料
with open('data/PubMed_ner_data.txt', 'r') as f:  # test
    data = f.readlines()

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')

# 定義方法來處理資料
def get_data(data):
    articles = {}
    labels = []
    for line_data in data:
        parts = line_data.strip().split('|')

        if len(parts) == 3:
            pmid, section, text = parts
            pmid = pmid.strip()
            section = section.strip()
            text = text.strip()
            if pmid not in articles:
                articles[pmid] = {'t': '', 'a': ''}
            articles[pmid][section] = text
        elif len(parts) == 5:
            pmid, start, end, name, biotype = parts
            pmid = pmid.strip()
            start = int(start.strip())
            end = int(end.strip())
            name = name.strip()
            biotype = biotype.strip()
            labels.append({
                'pmid': pmid,
                'start': start,
                'end': end,
                'name': name,
                'biotype': biotype
            })

    return articles, labels

# 直接進行分詞和 BIO 標註對齊
def tokenize_and_label(text, labels, tokenizer):
    bio_labels = ['O'] * len(text)

    # 對文章進行 BIO 標註
    for label in labels:
        start, end = label['start'], label['end']
        if start < len(text) and end <= len(text):
            bio_labels[start] = f'B-{label["biotype"]}'
            for i in range(start + 1, end):
                bio_labels[i] = f'I-{label["biotype"]}'

    tokens = []
    token_labels = []
    char_index = 0

    words = text.split()
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        for i, token in enumerate(word_tokens):
            if char_index < len(bio_labels):
                token_label = bio_labels[char_index]
                if i > 0 and token_label.startswith('B-'):
                    token_label = 'I-' + token_label[2:]
                tokens.append(token)
                token_labels.append(token_label)
        char_index += len(word) + 1  # 加1來跳過空格

    return tokens, token_labels

# 將 BIO 標註轉換為標籤編號
def convert_labels_to_ids(token_labels, label_to_id_map):
    label_ids = [label_map.get(label, 0) for label in token_labels]
    return label_ids

# 多線程處理每篇文章
def process_article(pmid, content, labels, tokenizer, label_map):
    try:
        full_text = content['t'] + " " + content['a']
        relevant_labels = [label for label in labels if label['pmid'] == pmid]

        # 直接使用 tokenizer 進行分詞並對齊標註
        tokens, token_bio_labels = tokenize_and_label(full_text, relevant_labels, tokenizer)

        # 使用 label_map 將 BIO 標註轉換為標籤編號
        label_ids = convert_labels_to_ids(token_bio_labels, label_map)

        # 將 token 和 label_id 分別組合成矩陣
        token_matrix = tokens  # 這裡 tokens 本身已經是一個 token 的序列
        label_matrix = label_ids  # label_ids 本身已經是一個標籤的序列

        return [pmid, token_matrix, label_matrix, token_bio_labels]

    except Exception as e:
        print(f"Error processing PMID {pmid}: {e}")
        return None

# 取得資料和標籤
articles, labels = get_data(data)

# 準備儲存結果的列表
all_data = []

# 設定最大線程數量
max_workers = 20

# 使用多線程處理文章
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_article, pmid, content, labels, tokenizer, label_map): pmid for pmid, content in articles.items()}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Articles"):
        result = future.result()
        if result:
            all_data.append(result)

# 隨機選取一筆資料來顯示 token, bio_label, 和 label_mapping 的資訊
if all_data:
    random_entry = random.choice(all_data)
    pmid, tokens, label_ids, token_bio_labels = random_entry

    # 顯示隨機選取的一筆資料的部分資訊
    print("PMID:", pmid)
    for token, bio_label, label_id in zip(tokens[:10], token_bio_labels[:10], label_ids[:10]):  # 只顯示前10個token
        print(f"Token: {token}, BIO Label: {bio_label}, Label ID: {label_id}")

# 將所有資料一次性儲存，包括Token_BIO_Labels
df = pd.DataFrame(all_data, columns=['PMID', 'Token_Matrix', 'Label_Matrix', 'Token_BIO_Labels'])
df.to_csv('data/PubMed_ner_data.csv', index=False)

print("資料已儲存到 CSV 檔案中")
