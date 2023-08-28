# 从目录中加载原始数据
from os import walk
import pandas as pd
import time
import faiss
import os

# 计算时间差
startTime = time.time()
dataDir = "/Users/lijunjun/PycharmProjects/faiss-demo/example01/data/input"
allFiles = next(walk(dataDir), (None, None, []))[2]
frames = []
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    frames.append(pd.read_csv("/Users/lijunjun/PycharmProjects/faiss-demo/example01/data/input/"+file, sep="`",header=None, names=["sentence"]))
df = pd.concat(frames, axis=0, ignore_index=True)
print("载入原始数据完毕，数据量", len(df))
index = faiss.read_index('my.index')
# 尝试进行查询
from sentence_transformers import SentenceTransformer
import os 
if os.path.exists('uer_sbert-base-chinese-nli'):
    model = SentenceTransformer('/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli')
else:
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")
print("载入模型完毕")

topK = 5
search = model.encode(["半导体"])
D, I = index.search(search, topK)
for i in range(I.shape[1]):
    result_index = I[0, i]
    result_score = D[0, i]
    result_sentence = df['sentence'].iloc[result_index]  # 这里需要根据你的数据结构进行调整
    print(f"相似度排名 #{i + 1}: 分数 {result_score}, 句子: {result_sentence}")
# ret = df['sentence'].iloc[I[0]]
# print(ret)
print("查询完毕，耗时", time.time() - startTime, "秒")