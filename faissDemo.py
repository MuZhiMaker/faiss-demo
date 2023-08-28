import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    # 使用pandas读取数据
    # df = pd.read_csv("./ready.txt", sep="#", header=None, names=["sentence"])
    df = pd.read_csv("data02.txt", sep="#", header=None, names=["sentence"])
    print(df)

    model = SentenceTransformer('uer/sbert-base-chinese-nli')
    # model.save("/Users/lijunjun/PycharmProjects/faiss-demo/uer_sbert-base-chinese-nli")
    sentences = df['sentence'].tolist()
    sentence_embeddings = model.encode(sentences)
    print(sentence_embeddings.shape)

    # 建立第一堆向量数据
    dimension = sentence_embeddings.shape[1]
    print("dimension:" + dimension)
    # 建立一个空的索引容器
    # IndexFlatL2是faiss库中最简单的索引，适用于小数据量的索引，它使用了精确的L2距离计算
    index = faiss.IndexFlatL2(dimension)
    # nlist = 50
    # quantizer = faiss.IndexFlatL2(dimension)
    # # IndexIVFFlat是faiss库中的索引，适用于大数据量的索引，它使用了精确的L2距离计算
    # # METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
    # index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    # # 倒排表索引类型需要训练
    # assert not index.is_trained  
    # index.train(sentence_embeddings)
    # # 处理好的向量数据灌入这个索引容器中
    # index.add(sentence_embeddings)

    # print("查看索引", index.ntotal)
    
    # 读取查询关键字文件
    with open('search00.txt', 'r', encoding='utf-8') as f:
        query_keywords = [line.strip() for line in f.readlines()]
        print(query_keywords)

    # 存储查询结果
    results = []
    
    # topK 定义了我们要查找多少条最相似的数据
    topK = 5
    # 循环处理查询关键字
    for query in query_keywords:
        print(f"查询关键字:{query}")
        
        result_info = []
        result_info.append(f"查询关键字: {query}")
        
        # 将要搜索的内容编码为向量
        search = model.encode([query])
        D, I = index.search(search, topK)
        
        # print('end', df['sentence'].iloc[I[0]])
        for i in range(I.shape[1]):
            result_index = I[0, i]
            result_score = D[0, i]
            if result_score > 200:
                continue
            result_sentence = df['sentence'].iloc[result_index]  # 这里需要根据你的数据结构进行调整
            result_info.append(f"相似度排名 #{i + 1}: 分数 {result_score}, 句子: {result_sentence}")
        result_info.append("---------------------")
        # result_info.append(str(df['sentence'].iloc[I[0]]))
        
        results.append("\n".join(result_info))
        
    # 将查询结果写入 result.txt 文件
with open("result.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(results))
