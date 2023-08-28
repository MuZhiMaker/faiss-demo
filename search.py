# 从目录中加载原始数据
from os import walk
import pandas as pd

dataDir = "./data/output"
allFiles = next(walk(dataDir), (None, None, []))[2]
frames = []
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    frames.append(pd.read_csv("./data/output/"+file, sep="`",header=None, names=["sentence"]))
df = pd.concat(frames, axis=0, ignore_index=True)
print("载入原始数据完毕，数据量", len(df))

# 加载预处理数据
import numpy as np
sentences = df['sentence'].tolist()
sentence_embeddings = np.load("data.npy")
print("载入向量数据完毕，数据量", len(sentence_embeddings))

# 使用构建向量时的模型来构建向量索引
import faiss
dimension = sentence_embeddings.shape[1]

# 使用不同的查询方式
myMode = '2'
if myMode == '1':
    # 使用L2距离计算
    index = faiss.IndexFlatL2(dimension)
elif myMode == '2':
    quantizer = faiss.IndexFlatL2(dimension)
    nlist = 50
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    index.train(sentence_embeddings)
    index.add(sentence_embeddings)
    print("建立向量索引完毕，数据量", index.ntotal)

# 尝试进行查询
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('uer/sbert-base-chinese-nli')
print("载入模型完毕")

topK = 5
search = model.encode(["整流罩"])
D, I = index.search(search, topK)
for i in range(I.shape[1]):
    result_index = I[0, i]
    result_score = D[0, i]
    result_sentence = df['sentence'].iloc[result_index]  # 这里需要根据你的数据结构进行调整
    print(f"相似度排名 #{i + 1}: 分数 {result_score}, 句子: {result_sentence}")
# ret = df['sentence'].iloc[I[0]]
# print(ret)