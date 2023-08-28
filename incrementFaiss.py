import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    # 使用pandas读取数据
    # df = pd.read_csv("./ready.txt", sep="#", header=None, names=["sentence"])
    df = pd.read_csv("origin.txt", sep="#", header=None, names=["sentence"])
    # print(df)

    # model = SentenceTransformer('uer_sbert-base-chinese-nli')
    # model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")
    dimension = 0
    if os.path.exists('uer_sbert-base-chinese-nli'):
        model = SentenceTransformer('/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli')
        dimension = model.get_sentence_embedding_dimension()
    else:
        model = SentenceTransformer('uer_sbert-base-chinese-nli')
        model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")
    sentences = df['sentence'].tolist()
    ids = df.index.values.tolist()
    sentence_embeddings = model.encode(sentences)
    # print(sentence_embeddings.shape)

    # 建立第一堆向量数据
    if dimension == 0:
        dimension = sentence_embeddings.shape[1]
    # 建立一个空的索引容器
    if os.path.exists('my.index'):
        index = faiss.read_index('my.index')
    else:
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)
        faiss.write_index(index, 'my.index')
    # 处理好的向量数据灌入这个索引容器中
    # index.add(sentence_embeddings)
    index.add_with_ids(sentence_embeddings, np.array(ids).astype('int64'))

    df2 = pd.read_csv("increment.txt", sep="#", header=None, names=["sentence"])
    add_sentence_embeddings = model.encode(df['sentence'].tolist()[(index.ntotal-1)-df2.index.values[-1]:])
    print('add_sentence_embeddings',add_sentence_embeddings)
    # add_id = df.index.values[-1] + 1
    add_total = df2.index.values[-1] - index.ntotal + 1
    print('add_total',add_total)
    add_id = []
    print('index.ntotal',index.ntotal)
    for i in range(add_total):
        add_id.append(index.ntotal + i)
    print('add_id',add_id)
    index.add_with_ids(add_sentence_embeddings, np.array(add_id).astype('int64'))
    print("查看索引", index.ntotal)

    # topK 定义了我们要查找多少条最相似的数据
    topK = 5
    # 将要搜索的内容编码为向量
    search = model.encode(["人工智能"])
    D, I = index.search(search, topK)
    print('end', df2['sentence'].iloc[I[0]])