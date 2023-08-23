import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    # 使用pandas读取数据
    # df = pd.read_csv("./ready.txt", sep="#", header=None, names=["sentence"])
    df = pd.read_csv("data.txt", sep="#", header=None, names=["sentence"])
    print(df)

    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")
    sentences = df['sentence'].tolist()
    sentence_embeddings = model.encode(sentences)
    print(sentence_embeddings.shape)

    # 建立第一堆向量数据
    dimension = sentence_embeddings.shape[1]
    # 建立一个空的索引容器
    index = faiss.IndexFlatL2(dimension)
    # 处理好的向量数据灌入这个索引容器中
    index.add(sentence_embeddings)

    print("查看索引", index.ntotal)

    # topK 定义了我们要查找多少条最相似的数据
    topK = 5
    # 将要搜索的内容编码为向量
    search = model.encode(["颜色"])
    D, I = index.search(search, topK)
    print('end', df['sentence'].iloc[I[0]])