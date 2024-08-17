import warnings
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import random
import numpy as np
from tqdm import tqdm
import os
import pickle
import yaml
from label import label_map

warnings.filterwarnings('ignore')

# 读取配置文件
with open("ner/settings.yml", 'r') as file:
    settings = yaml.safe_load(file)['ner']

# 设置随机种子
seed = settings['seed']
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

BATCH_SIZE = settings['batch_size']
TRAIN_SPLIT = settings['train_split']
MAX_LEN = settings['max_len']
file_path = settings['data_file_path']
tokenized_data_path = settings['tokenized_data_path']
model_name = settings['model_name']

class NERDataset(Dataset):
    def __init__(self, file_path, tokenized_data_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.articles = {}
        self.labels = []
        self.data = []

        if tokenized_data_path and os.path.exists(tokenized_data_path):
            print("Loading tokenized data...")
            with open(tokenized_data_path, 'rb') as file:
                self.data = pickle.load(file)
        else:
            print("Reading data file...")
            with open(file_path, 'r', encoding='utf-8') as file:
                data = file.readlines()

            print("Processing data lines...")
            for line_data in tqdm(data, desc="Processing data lines"):
                parts = line_data.strip().split('|')

                # 检查格式是否正确
                if len(parts) == 3 or len(parts) == 5:
                    if len(parts) == 3:
                        pmid, section, text = parts
                        pmid = pmid.strip()
                        section = section.strip()
                        text = text.strip()
                        if pmid not in self.articles:
                            self.articles[pmid] = {'t': '', 'a': ''}
                        self.articles[pmid][section] = text
                    elif len(parts) == 5:
                        pmid, start, end, name, biotype = parts
                        pmid = pmid.strip()
                        start = int(start.strip())
                        end = int(end.strip())
                        name = name.strip()
                        biotype = biotype.strip()
                        self.labels.append({
                            'pmid': pmid,
                            'start': start,
                            'end': end,
                            'name': name,
                            'biotype': biotype
                        })

            for pmid in self.articles:
                self.articles[pmid] = self.articles[pmid]['t'] + ' ' + self.articles[pmid]['a']

            print(f"Total articles: {len(self.articles)}")
            self._create_dataset()

            if tokenized_data_path:
                with open(tokenized_data_path, 'wb') as file:
                    pickle.dump(self.data, file)

    def _create_dataset(self):
        articles_list = list(self.articles.items())

        print("\nProcessing all articles...")
        for article in tqdm(articles_list, desc="Tokenizing and labeling"):
            result = self._process_article_safe(article)
            if result:
                self.data.append(result)

    def _process_article_safe(self, article):
        try:
            return self._process_article(article)
        except Exception as e:
            print(f"Error processing article {article[0]}: {e}")
            return None

    def _process_article(self, article):
        pmid, text = article
        char_bio_labels = self._create_char_bio_labels(text, [label for label in self.labels if label['pmid'] == pmid])
        tokens, token_labels = self._tokenize_and_label(text, char_bio_labels)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [self._label_to_id(label) for label in token_labels]

        # 截断和填充序列
        token_ids = token_ids[:MAX_LEN] + [0] * (MAX_LEN - len(token_ids))
        label_ids = label_ids[:MAX_LEN] + [0] * (MAX_LEN - len(label_ids))
        token_labels = token_labels[:MAX_LEN] + ['O'] * (MAX_LEN - len(token_labels))

        # 实时打印每一篇文章的处理结果
        print(f"\nArticle PMID: {pmid}")
        print("Token IDs:")
        print(token_ids)  # 打印token的ID矩阵
        print("Label Mapping IDs:")
        print(label_ids)  # 打印label ID的矩阵
        print("Token Labels:")
        print(token_labels)  # 打印label的字符串矩阵

        return torch.tensor(token_ids), torch.tensor(label_ids)

    def _create_char_bio_labels(self, text, labels):
        bio_labels = ['O'] * len(text)
        for label in labels:
            start, end = label['start'], label['end']
            if start < len(text) and end <= len(text):
                bio_labels[start] = f'B-{label["biotype"]}'
                for i in range(start + 1, end):
                    if bio_labels[i] != 'O':  # 如果已经有标签存在，跳过避免覆盖
                        continue
                    bio_labels[i] = f'I-{label["biotype"]}'
        return bio_labels
    
    def _tokenize_and_label(self, text, char_bio_labels):
        tokens = []
        token_labels = []
        word_start_idx = 0

        for word in text.split():
            word_tokens = self.tokenizer.tokenize(word)
            word_label = char_bio_labels[word_start_idx]  # 获取单词的第一个字母的标签
            for i, token in enumerate(word_tokens):
                if i > 0:
                    # 确保在单词内的子词使用 I- 标签
                    if word_label.startswith('B-'):
                        word_label = 'I-' + word_label[2:]
                tokens.append(token)
                token_labels.append(word_label)
            word_start_idx += len(word) + 1  # 更新下一个单词的起始索引

        return tokens, token_labels

    def _label_to_id(self, label):
        return label_map.get(label, 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def collate_fn(batch):
    token_ids, label_ids = zip(*batch)
    max_len = max(len(ids) for ids in token_ids)
    padded_token_ids = [torch.cat([ids, torch.tensor([0] * (max_len - len(ids)))]) for ids in token_ids]
    padded_label_ids = [torch.cat([ids, torch.tensor([0] * (max_len - len(ids)))]) for ids in label_ids]
    return torch.stack(padded_token_ids), torch.stack(padded_label_ids)

# 创建数据集和数据加载器
dataset = NERDataset(file_path, tokenized_data_path)
train_size = int(TRAIN_SPLIT * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(seed))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 打印训练集和验证集的大小
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
