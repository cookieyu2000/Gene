import pandas as pd

# # 讀取 Excel 檔案
# data = pd.read_excel('data/HGMD_ann.xlsx')

# # 將資料儲存為 HDF5 格式
# data.to_hdf('data/HGMD_ann.h5', key='df', mode='w')

# 讀取 HDF5 格式的資料
data = pd.read_hdf('data/HGMD_ann.h5', key='df')

# 過濾 HGMD_class 欄位，只保留 "DM" 和 "DM?"
filter_data = data[data['HGMD_class'].isin(['DM', 'DM?'])]

# 將 HGMD_pubmed 欄位中的值以分號拆分成多列
split_data = filter_data['HGMD_pubmed'].str.split(',', expand=True)

# 將過濾後的 HGMD_class 和拆分後的 HGMD_pubmed 欄位組合在一起
result_data = filter_data[['HGMD_class']].join(split_data[0].rename('HGMD_pubmed')).dropna()

result_data.to_csv('data/HGMD_pubmed.csv', index=False)