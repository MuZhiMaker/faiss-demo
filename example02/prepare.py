import os
from os import walk
import pandas as pd
import faiss

dataDir = "/Users/lijunjun/PycharmProjects/faiss-demo/example01/data/input"
allFiles = next(walk(dataDir), (None, None, []))[2]
print("载入原始数据完毕，数据量", len(allFiles))
# 加载原始数据
frames = []
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    frames.append(pd.read_csv("/Users/lijunjun/PycharmProjects/faiss-demo/example01/data/input/"+file, sep="`",header=None, names=["sentence"]))
df = pd.concat(frames, axis=0, ignore_index=True)

# 加载模型，将数据进行向量化处理
from sentence_transformers import SentenceTransformer
if os.path.exists('uer_sbert-base-chinese-nli'):
    model = SentenceTransformer('/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli')
else:
    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")

sentences = df['sentence'].tolist()
sentence_embeddings = model.encode(sentences)
dimension = sentence_embeddings.shape[1]
# 使用不同的查询方式
myMode = '2'
if myMode == '1':
    # 使用L2距离计算
    index = faiss.IndexFlatL2(dimension)
elif myMode == '2':
    if os.path.exists('my.index'):
        index = faiss.read_index('my.index')
    else:
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 50, faiss.METRIC_L2)
        index.train(sentence_embeddings)
        index.add(sentence_embeddings)
        faiss.write_index(index, 'my.index')
        
print("prepare 结束")