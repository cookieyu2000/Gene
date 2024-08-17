# 假設你的檔案路徑是 'data.txt'
file_path = 'data/tmVar3.PubMed_ner.txt'

# 打開檔案並逐行讀取
with open(file_path, 'r') as file:
    lines = file.readlines()

# 計算 "| t |" 出現的次數
count = sum(1 for line in lines if "| t |" in line)

print(f'出現了 {count} 次 "| t |"')
