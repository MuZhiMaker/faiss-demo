import flask, json
from flask import request
import time
from os import walk
import pandas as pd
import faiss
 
'''
flask： web框架，通过flask提供的装饰器@server.route()将普通函数转换为服务
登录接口，需要传url、username、passwd
'''
# 创建一个服务，把当前这个python文件当做一个服务
server = flask.Flask(__name__)
# server.config['JSON_AS_ASCII'] = False
# @server.route()可以将普通函数转变为服务 登录接口的路径、请求方式
@server.route('/login', methods=['get', 'post'])
def login():
    # 获取通过url请求传参的数据
    username = request.values.get('name')
    # 获取url请求传的密码，明文
    pwd = request.values.get('pwd')
    # 判断用户名、密码都不为空，如果不传用户名、密码则username和pwd为None
    if username and pwd:
        if username=='xiaoming' and pwd=='111':
            resu = {'code': 200, 'message': '登录成功'}
            return json.dumps(resu, ensure_ascii=False)  # 将字典转换为json串, json是字符串
        else:
            resu = {'code': -1, 'message': '账号密码错误'}
            return json.dumps(resu, ensure_ascii=False)
    else:
        resu = {'code': 10001, 'message': '参数不能为空！'}
        return json.dumps(resu, ensure_ascii=False)
    
@server.route('/search', methods=['get', 'post'])
def search():
    # 获取通过url请求传参的数据
    query = request.values.get('query')
    if query:
        # 计算时间差
        wholeStartTime = time.time()
        startTime = time.time()
        dataDir = "/Users/lijunjun/PycharmProjects/faiss-demo/example02/data/input"
        allFiles = next(walk(dataDir), (None, None, []))[2]
        frames = []
        for i in range(len(allFiles)):
            file = allFiles[i]
            print(file)
            frames.append(pd.read_csv("/Users/lijunjun/PycharmProjects/faiss-demo/example02/data/input/"+file, sep="`",header=None, names=["sentence"]))
        df = pd.concat(frames, axis=0, ignore_index=True)
        print("载入原始数据完毕，数据量", len(df))
        print("载入原始数据完毕，耗时", time.time() - startTime, "秒")
        startTime = time.time()
        index = faiss.read_index('/Users/lijunjun/PycharmProjects/faiss-demo/example02/my.index')
        print("载入索引完毕，耗时", time.time() - startTime, "秒")
        # 尝试进行查询
        from sentence_transformers import SentenceTransformer
        import os 
        startTime = time.time()
        if os.path.exists('/Users/lijunjun/PycharmProjects/faiss-demo/example02/uer_sbert-base-chinese-nli'):
            print("本地存在模型")
            model = SentenceTransformer('/Users/lijunjun/PycharmProjects/faiss-demo/example02/uer_sbert-base-chinese-nli')
        else:
            print("本地不存在模型")
            model = SentenceTransformer('uer/sbert-base-chinese-nli')
            model.save("/Users/lijunjun/PycharmProjects/faiss-demo/example02/uer_sbert-base-chinese-nli")
        print("载入模型完毕")
        print("载入模型完毕，耗时", time.time() - startTime, "秒")

        # 存储查询结果
        results = []
        topK = 5

        startTime = time.time()
        startTime = time.time()
        # 循环处理查询关键字
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
            result_sentence = df['sentence'].iloc[result_index]  # 这里需要根据你的数据结构进行调整
            result_info.append(f"相似度排名 #{i + 1}: 分数 {result_score}, 句子: {result_sentence}")
        result_info.append("---------------------")
        # result_info.append(str(df['sentence'].iloc[I[0]]))
        
        results.append("\n".join(result_info))
        print(results)
        print("查询完毕，耗时", time.time() - startTime, "秒")

        startTime = time.time()
        print("查询完毕，耗时", time.time() - wholeStartTime, "秒")
    
    resu = {'code': 200, 'message': result_info}
    return json.dumps(resu, ensure_ascii=False) 
 
if __name__ == '__main__':
    server.run(debug=True, port=8888, host='0.0.0.0')# 指定端口、host,0.0.0.0代表不管几个网卡，任何ip都可以访问