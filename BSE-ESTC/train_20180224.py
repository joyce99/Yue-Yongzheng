# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:11:25 2018

@author: lizheng
"""

from sklearn import metrics
import sys

sys.path.append("D:\PycharmProjects\Tax")

import spbm_model11 as sp
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import PassiveAggressiveClassifier
import datetime


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import csr_matrix

from gensim.models import word2vec
import gensim



import json
from urllib.parse import quote
import string
import requests
# import time


def get_data(filename):
    csvfile = open(filename, 'rb')
    reader = pd.read_csv(csvfile)
    reader = reader.append(reader)
     #分割商品名称
    #reader = reader.drop_duplicates()

    reader['HWMC'] = sp.sjcl(list(reader['HWMC'].astype(str)))
    print("分词结束", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))



    # ##多线程分词
    # print("开始分词", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
    # # keywordextraction,
    # from multiprocessing import Pool  # 多线程处理机制，创建多个子进程
    # p = Pool(30)
    # reader['HWMC'] = p.map(sp.cut_jieba, list(reader['HWMC'].astype(str)))
    # p.close()
    # p.join()
    #
    # print("分词结束", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))



    # # ###将分词结果保存为csv格式
    # df = pd.DataFrame({'HWMC':reader['HWMC']})
    #
    # #API匹配分词结果
    # dfNew = pd.DataFrame({'HWMC':reader['HWMC']})
    # valuearr = df.values
    # rowIndex = 0
    # i = 0
    # for sentenceindex in valuearr:
    #     rowIndex += 1
    #     for wordInSentence in sentenceindex:
    #         arrWordToSimilar = wordInSentence.split(' ')
    #         for wordToCheck in arrWordToSimilar:
    #             # headers = {
    #             #     'Connection': 'close',
    #             # }
    #             url = 'http://shuyantech.com/api/cndbpedia/ment2ent?q='+wordToCheck+'&apikey=cd9ad381122aeade145b3c55258d9be7'
    #             # r = requests.get(quote(url, safe=string.printable), headers=headers)
    #             r = requests.get(quote(url, safe=string.printable))
    #             # r = requests.get(url, headers=headers)
    #             i += 1
    #             print(i)
    #             str0 = json.loads(r.text)
    #             rValue = str0['ret']
    #             if len(rValue) > 0:
    #                 str1 = ''.join(rValue[0])
    #                 if str1.find('（') != -1:
    #                     str2 = str1.split('（')[1].split('）')[0]
    #                     strIndfNew = dfNew.values[rowIndex-1][0]
    #                     strIndfNew = strIndfNew + ' ' + str2
    #                     dfNew.values[rowIndex-1][0] = strIndfNew
    # df = dfNew
    # df.to_csv('fencijieguo1.5-3k.csv', index=False, index_label=False, encoding='utf-8-sig')
    # reader['HWMC'] = dfNew



    reader['HWMC'] = reader['HWMC'].apply(lambda x: np.NaN if str(x)=='' else x) #将空白替换为nan
    reader = reader[reader['HWMC'].notnull()]
    reader = shuffle(reader)
    reader.index = np.arange(len(reader))
    classes = reader.drop_duplicates('U_CODE')
    all_classes = np.asarray(classes['U_CODE'])
    X = np.array(reader['HWMC'])
    y = np.array(reader['U_CODE'])
    csvfile.close()
    return X, y, all_classes


##转换为字典
def Dict(X):
    docs = X
    segments = []
    for i in range(len(X)):
        words = docs[i].split(' ')
        # words   = np.array(words)
        for j in range(len(words)):
            fileContent = words[j]
            segments.append(fileContent)
    ##去重
    segments =list(set(segments))
    return segments


##转换为list
def transfer(X):
    docs = X
    segments = []
    for i in range(len(X)):

        segments.append(docs[i])
    return segments

##



def tfidf_compute(word, doc):

    docs = np.array(doc)
    words = np.array(word)


    # 词在文档中出现的个数
    cfs = []
    for e in docs:
       cf = [e.count(word) for word in words]
       cfs.append(cf)
    # print('==============================================')
    # print('cfs:\n', np.array(cfs))


    # 词在文档中出现的频率
    tfs = []
    for e in cfs:
        tf = e/(np.sum(e))
        tfs.append(tf)
    # print('==============================================')
    # print('tfs:\n', np.array(tfs))


    # 包含词的文档个数
    dfs = list(np.zeros(words.size, dtype=int))
    for i in range(words.size):
        for doc in docs:
            if doc.find(words[i]) != -1:
                dfs[i] += 1
    # print('==============================================')
    # print('df:\n', np.array(dfs))


    # 计算每个词的idf(逆向文件频率inverse document frequency)
    # #log10(N/(1+DF))
    N = np.shape(docs)[0]
    # f(e) = np.log10(N*1.0/(1+e))
    idfs = [(np.log10(N*1.0/(1+e))) for e in dfs]

    # print('==============================================')
    # print('idfs:',np.array(idfs))


    # 计算tf-idf(term frequency - inverse document frequency)
    tfidfs = []
    for i in range(np.shape(docs)[0]):
        word_tfidf = np.multiply(tfs[i], idfs)
        tfidfs.append(word_tfidf)
    # print('==============================================')
    # print('tfidfs:\n', np.array(tfidfs))


def sklearn_tfidf(docs):
    tag_list = docs

    vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵

    X = vectorizer.fit_transform(tag_list)  # 计算个词语出现的次数

    transformer = TfidfTransformer()

    tfidf = transformer.fit_transform(X)  # 将词频矩阵X统计成TF-IDF值

    # print(tfidf)

    return tfidf



def sklearn_word2vec(docs):

    #加载数据
    sentences = docs

    #训练模型
    model = word2vec.Word2Vec(sentences, size=100, hs=1, min_count=1, window=3)

    return model
    # return csr_matrix(model)



def fit_form(model, data):

    martixs = model(data)

    print(martixs)

    return martixs

    # return csr_matrix(martixs)











if __name__ == "__main__":
    MD = PassiveAggressiveClassifier(max_iter=1000, loss='squared_hinge', average=10, n_jobs=-1)
    X, y, all_classes = get_data('traindata1.5k.csv')


    # # word = dict(X)
    doc = transfer(X)
    print(doc)

    Dict = Dict(X)
    print(Dict)



    # # tfidf_compute(word, doc)
    #

    model = sklearn_word2vec(Dict)
    print(model['前'])


    #
    # model.save('/tmp')
    #
    # # model.save('/tmp/MyModel')
    # # model.save_word2vec_format('/tmp/MyModel.txt', binary=False)
    # # model = gensim.models.Word2Vec.load('/tmp/MyModel')
    #
    # model = gensim.models.Word2Vec.load('/tmp')



    # sss = StratifiedShuffleSplit(n_splits = 10,test_size = 0.2)#训练集额和测试集的比例随机选定，训练集和测试集的比例的和可以小于1,但是还要保证训练集中各类所占的比例是一样的
    # sss.get_n_splits(X, y)
    # i = 0
    # for train_index, test_index in sss.split(X, y):
    #     print("{} time".format(i))
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #
    #
    #
    #     # MD.partial_fit(sklearn_word2vec(transfer(X_train)), y_train, classes = all_classes)
    #     # result = MD.predict(sklearn_word2vec(transfer(X_test)))
    #
    #
    #
    #     MD.partial_fit(sp.get_hv(X_train), y_train, classes = all_classes)
    #     result = MD.predict(sp.get_hv(X_test))
    #
    #
    #
    #
    #
    #
    #     print("正确率: %.4g" % metrics.accuracy_score(y_test, result), "召回率: %.4g" % metrics.recall_score(y_test, result, average='macro')
    #     , "F1: %.4g" % metrics.f1_score(y_test, result, average='weighted'))
    #     i += 1
    #
    # #joblib.dump(MD, "D:\\PycharmProjects\\Tax\\321_2**15.pkl.gz", compress=('gzip', 3))