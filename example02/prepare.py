import os
from os import walk
import pandas as pd
import faiss

# 当前文件所在的目录
current_work_dir = os.path.dirname(__file__)
weight_path=''
weight_path = os.path.join(current_work_dir, weight_path)

dataDir = os.path.join(current_work_dir, 'data/input')
# dataDir = "/Users/lijunjun/PycharmProjects/faiss-demo/example02/data/input"
allFiles = next(walk(dataDir), (None, None, []))[2]
print("载入原始数据完毕，数据量", len(allFiles))
# 加载原始数据
frames = []
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    frames.append(pd.read_csv(dataDir + '/' + file, sep="`",header=None, names=["sentence"]))
df = pd.concat(frames, axis=0, ignore_index=True)

# 加载模型，将数据进行向量化处理
from sentence_transformers import SentenceTransformer
uerSbertBaseChineseNliPath = os.path.join(current_work_dir, 'uer_sbert-base-chinese-nli')
if os.path.exists(uerSbertBaseChineseNliPath):
    model = SentenceTransformer(uerSbertBaseChineseNliPath)
else:
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    model.save(uerSbertBaseChineseNliPath)

sentences = df['sentence'].tolist()
sentence_embeddings = model.encode(sentences)
dimension = sentence_embeddings.shape[1]

indexPath = os.path.join(weight_path, 'my.index')
# 使用不同的查询方式
myMode = '2'
if myMode == '1':
    # 使用L2距离计算
    index = faiss.IndexFlatL2(dimension)
elif myMode == '2':
    if os.path.exists(indexPath):
        index = faiss.read_index(indexPath)
    else:
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 50, faiss.METRIC_L2)
        index.train(sentence_embeddings)
        index.add(sentence_embeddings)
        faiss.write_index(index, indexPath)
        
print("prepare 结束")