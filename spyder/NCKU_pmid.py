import json
import pandas as pd
import aiohttp
import asyncio
from tqdm.asyncio import tqdm as tqdm_asyncio  # 用于异步的 tqdm
from aiohttp import ClientTimeout

input_data = 'data/HGMD_pubmed.csv'
output_data = 'data/sentence_DM_not_2.txt'
MAX_CONCURRENT_REQUESTS = 1000  # 限制最大并发请求数量
MAX_RETRIES = 100  # 最大重试次数

def read_data(input_data):
    '''
    Read the input data
    '''
    df = pd.read_csv(input_data, usecols=['HGMD_class', 'HGMD_pubmed'])
    return df


async def call_api_async(session, id, retries=MAX_RETRIES):
    '''
    Call the NCBI API to fetch data for a given ID asynchronously
    '''
    api = f'https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids={id}'
    timeout = ClientTimeout(total=20)
    for attempt in range(retries):
        try:
            async with session.get(api, timeout=timeout) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    await asyncio.sleep(3)  # Rate limiting delay
                else:
                    print(f"Failed to fetch data for ID {id}, Status Code: {response.status}")
        except Exception as e:
            print(f"Exception occurred for ID {id}: {e}")
        await asyncio.sleep(2)  # 等待2秒后再重试
    return None  # 所有重试均失败

def process_response(id, response, hgmd_class_label):
    '''
    處理 API 回應並加入 hgmd_class_label 作為額外欄位
    '''
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        print(f"JSON Decode Error for ID {id}")
        return []

    results = []
    annotations = []
    valid_data = False
    for record in data.get("PubTator3", []):
        pmid = record.get("pmid", id)
        for passage in record.get("passages", []):
            infons = passage.get("infons", {})
            text = passage.get("text", "")
            if infons.get("type") == "title" and text.strip():
                results.append(f"{pmid}| t |{text}")
                valid_data = True
            elif infons.get("type") == "abstract" and text.strip():
                results.append(f"{pmid}| a |{text}")
                valid_data = True
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
                    except ValueError:
                        continue

    if valid_data:
        # 添加 hgmd_class_label 行
        results.append(f"{pmid}| l |{hgmd_class_label}")
        results.extend(annotations)
    return results

async def process_id(semaphore, session, row, file_handle, pbar, failed_ids):
    async with semaphore:  # 使用信号量限制并发请求数
        id = row.HGMD_pubmed
        hgmd_class_label = row.HGMD_class
        response = await call_api_async(session, id)
        if response:
            results = process_response(id, response, hgmd_class_label)
            if results:
                for result in results:
                    file_handle.write(result + '\n')
                file_handle.write("\n")
                file_handle.flush()  # 立即刷新文件缓冲区
        else:
            failed_ids.append(id)  # 记录失败的 ID
        pbar.update(1)  # 每处理完一个任务，更新进度条

async def process_ids_async(df_id, output_file):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)  # 限制并发请求数
    failed_ids = []

    async with aiohttp.ClientSession() as session:
        with open(output_file, 'w', encoding='utf-8') as file_handle:
            tasks = []
            with tqdm_asyncio(total=len(df_id), desc="Processing IDs") as pbar:
                for row in df_id.itertuples(index=False):
                    task = process_id(semaphore, session, row, file_handle, pbar, failed_ids)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)  # 等待所有任务完成

    if failed_ids:
        print(f"Retrying {len(failed_ids)} failed IDs...")
        async with aiohttp.ClientSession() as session:
            with open(output_file, 'a', encoding='utf-8') as file_handle:  # 追加模式打开文件
                tasks = []
                with tqdm_asyncio(total=len(failed_ids), desc="Retrying Failed IDs") as pbar:
                    for id in failed_ids:
                        row = df_id[df_id['HGMD_pubmed'] == id].iloc[0]
                        task = process_id(semaphore, session, row, file_handle, pbar, failed_ids=[])
                        tasks.append(task)
                    
                    await asyncio.gather(*tasks)  # 再次尝试所有失败的任务

# 讀取數據並去重
df_id = read_data(input_data)
df_id = df_id.drop_duplicates(subset=['HGMD_pubmed'])

# 將 HGMD_class 列轉換為 1 和 0
df_id['HGMD_class'] = df_id['HGMD_class'].apply(lambda x: 1 if x == 'DM' else 0)

# 異步處理每個 ID 並顯示進度條，同時同步寫入文件
asyncio.run(process_ids_async(df_id, output_data))

print(f"Processing completed and data saved to '{output_data}'")


# 38143