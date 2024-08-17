import json
import pandas as pd
import requests
from tqdm import tqdm
import time
import os

input_data = 'data/pub_cls.csv'
output_data = 'data/PubMed_ner_data.txt'
error_data = 'error/error_pmids.txt'  # 存放出錯的PMID

def read_data(input_data):
    '''
    讀取輸入資料
    '''
    df = pd.read_csv(input_data, usecols=['ID'])
    print(f"Read {len(df)} IDs from '{input_data}'")
    return df

def call_api(id):
    '''
    同步調用 NCBI API 以獲取給定 ID 的資料
    '''
    api = f'https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={id}'
    for _ in range(10):
        try:
            response = requests.get(api, timeout=20)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                time.sleep(3)  # 遇到速率限制時的延遲
            else:
                return None
        except requests.exceptions.RequestException:
            continue
    return None

def process_response(id, response):
    '''
    處理 API 回應
    '''
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return None

    results = []
    annotations = []
    has_title = False
    has_abstract = False
    has_annotations = False

    for record in data.get("PubTator3", []):
        pmid = record.get("pmid", id)
        for passage in record.get("passages", []):
            infons = passage.get("infons", {})
            text = passage.get("text", "")
            if infons.get("type") == "title" and text.strip():
                results.append(f"{pmid}| t |{text}")
                has_title = True
            elif infons.get("type") == "abstract" and text.strip():
                results.append(f"{pmid}| a |{text}")
                has_abstract = True
            for annotation in passage.get("annotations", []):
                offset = annotation.get("locations", [{}])[0].get("offset", "")
                length = annotation.get("locations", [{}])[0].get("length", "")
                name = annotation.get("text", "")
                biotype = annotation.get("infons", {}).get("biotype", "")
                if offset and length and name and biotype:
                    try:
                        offset = int(offset)
                        length = int(length)
                        end_offset = offset + length
                        annotations.append(f"{pmid} | {offset} | {end_offset} | {name} | {biotype}")
                        has_annotations = True
                    except ValueError:
                        continue

    if has_title and has_abstract and has_annotations:
        results.extend(annotations)
        return results
    else:
        return None  # 如果缺少任何一部分則返回 None

def process_id(id, file_handle, error_handle):
    '''
    處理單個 ID 的資料
    '''
    response = call_api(id)
    if response:
        results = process_response(id, response)
        if results:
            for result in results:
                file_handle.write(result + '\n')
            file_handle.write("\n")
            file_handle.flush()  # 立即寫入文件
        else:
            error_handle.write(f"{id}\n")  # 若資料不完整，記錄 PMIDs
            error_handle.flush()
    else:
        error_handle.write(f"{id}\n")  # 若 API 請求失敗，記錄 PMIDs
        error_handle.flush()

def main():
    # 創建錯誤資料夾（如果不存在）
    os.makedirs(os.path.dirname(error_data), exist_ok=True)

    df_id = read_data(input_data)
    id_list = df_id['ID'].tolist()
    
    with open(output_data, 'a', encoding='utf-8') as file_handle, \
         open(error_data, 'a', encoding='utf-8') as error_handle:  # 使用 'a' 模式以追加方式寫入
        for id in tqdm(id_list, desc="Processing IDs"):
            process_id(id, file_handle, error_handle)

# 同步處理每個 ID 並顯示進度條，同時同步寫入文件
main()

print(f"Processing completed and data saved to '{output_data}'")
print(f"Errors (if any) are logged in '{error_data}'")
