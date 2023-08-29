# 从目录中加载原始数据
from os import walk
import pandas as pd
import time
import faiss

import os
# 当前文件所在的目录
current_work_dir = os.path.dirname(__file__)
weight_path=''
weight_path = os.path.join(current_work_dir, weight_path)

# 计算时间差
wholeStartTime = time.time()
startTime = time.time()
dataDir='data/input'
dataDir = os.path.join(weight_path, dataDir)
allFiles = next(walk(dataDir), (None, None, []))[2]
frames = []
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    frames.append(pd.read_csv(dataDir + '/' + file, sep="`",header=None, names=["sentence"]))
df = pd.concat(frames, axis=0, ignore_index=True)
print("载入原始数据完毕，数据量", len(df))
print("载入原始数据完毕，耗时", time.time() - startTime, "秒")
startTime = time.time()
indexPath = os.path.join(weight_path, 'my.index')
index = faiss.read_index(indexPath)
print("载入索引完毕，耗时", time.time() - startTime, "秒")
# 尝试进行查询
from sentence_transformers import SentenceTransformer
import os 
startTime = time.time()
uerSbertBaseChineseNliPath = os.path.join(current_work_dir, 'uer_sbert-base-chinese-nli')
if os.path.exists(uerSbertBaseChineseNliPath):
    model = SentenceTransformer(uerSbertBaseChineseNliPath)
else:
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    model.save(uerSbertBaseChineseNliPath)
print("载入模型完毕")
print("载入模型完毕，耗时", time.time() - startTime, "秒")

# 存储查询结果
results = []
topK = 5

startTime = time.time()
# 读取查询关键字文件
searchPath = os.path.join(weight_path, 'search.txt')
with open(searchPath, 'r', encoding='utf-8') as f:
    query_keywords = [line.strip() for line in f.readlines()]
print("载入查询关键字完毕，耗时", time.time() - startTime, "秒")

startTime = time.time()
# 循环处理查询关键字
for query in query_keywords:
    print(f"查询关键字:{query}")
    
    result_info = []
    result_info.append(f"查询关键字: {query}")
    
    # 将要搜索的内容编码为向量
    search = model.encode([query])
    D, I = index.search(search, topK)
    
    for i in range(I.shape[1]):
        result_index = I[0, i]
        result_score = D[0, i]
        result_sentence = df['sentence'].iloc[result_index]  # 这里需要根据你的数据结构进行调整
        result_info.append(f"相似度排名 #{i + 1}: 分数 {result_score}, 句子: {result_sentence}")
    result_info.append("---------------------")
    
    results.append("\n".join(result_info))
print("查询完毕，耗时", time.time() - startTime, "秒")

startTime = time.time()
# 将查询结果写入 result.txt 文件
resultPath = os.path.join(weight_path, 'data/output/result.txt')
with open(resultPath, "w", encoding="utf-8") as f:
    f.write("\n\n".join(results))
print("写入文件完毕，耗时", time.time() - startTime, "秒")
print("查询完毕，耗时", time.time() - wholeStartTime, "秒")

