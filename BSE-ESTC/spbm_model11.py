# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:11:25 2018

@author: lizheng
"""

import jieba
#import jieba_fast.analyse
jieba.load_userdict('mix ciku.txt')
stopwords = [line.rstrip() for line in open('stopword.txt', mode='r', encoding='UTF-8')]
import pandas as pd


##原始分词
def sjcl(x):
    '''
    jieba分词
    '''
    import datetime
    print("开始分词", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))

    import jieba.analyse
    # jieba.enable_parallel(4)  # 并行分词 仅用于linux系统
    # jieba.load_userdict('ciku.txt')


    '''
    jieba中文分词
    '''
    import re
    # 匹配中文的分词
    # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')

    # 开始分词，对商品名称进行切割
    train_X = []
    for i in range(len(x)):
        num = 0
        segments = []
        fileContent = x[i]
        # segs = jieba.cut(fileContent)#全模式
        segs = jieba.cut_for_search(fileContent)#搜索引擎模式
        # segs = jieba.cut(fileContent, cut_all=False)#精准模式
        for seg in segs:
            if zhPattern.search(seg):
                if seg not in stopwords:
                    segments.append(seg)
                num += 1
        if num > 2:
            fileContent = ' '.join(segments)
        else:
            fileContent = ' '.join(segments + segments)
        train_X.append(fileContent)
    return train_X





# ##税收编码表_final + 分词结果同义词分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#
#     print("开始分词",datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#
#     import jieba
#
#     #import jieba_fast as jieba
#     #import jieba_fast.analyse
#
#     import jieba.analyse
#     jieba.load_userdict('ciku.txt')
#     #jieba.enable_parallel(4)#并行分词 仅用于linux系统
#
#     stopwords = [line.rstrip() for line in open('stopword.txt','r', encoding='UTF-8')]
#
#     '''
#     jieba中文分词
#     '''
#     import re
#     #匹配中文的分词
#     #zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     #开始分词，对商品名称进行切割
#     train_X = []
#
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         # segs = jieba.cut(fileContent)#全模式
#         # segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         segs = jieba.cut(fileContent, cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#
#                     segments.append(seg)
#                     synonyms = match(seg)
#                     if synonyms != -1:
#                         segments.append(synonyms)
#                     Synonyms = match_synonyms('税收编码表_final.txt')
#                     for i in range(len(Syno nyms)):
#                         Synonym = Synonyms[i].split(" ")
#                         for j in range(len(Synonym)):
#                             if seg == Synonym[j]:
#                                 Synony = Synonyms[i].split("\n")
#                                 segments.append(Synony[0])
#
#                     segments.append(seg)
#                 num += 1
#         if num>2:
#             fileContent = " ".join(segments)
#         else:
#             fileContent = " ".join(segments+segments)
#         train_X.append(fileContent)
#     return train_X
#
#
#
# ##匹配税收编码表_final同义词
# def match_synonyms(filename):
#     f = open(filename, mode='r', encoding='utf-8')
#     lines = f.readlines()
#     AnswerWord = []
#     for line in lines:
#         strRemovalDuplicate = ''
#         arrWord = line.split(" ")
#         for i in range(len(arrWord)):
#             # print(i, arrWord[i])
#             if i == 0:
#                 strRemovalDuplicate += arrWord[i]
#                 strRemovalDuplicate += ' '
#             else:
#                 if arrWord[i] != arrWord[0]:
#                     strRemovalDuplicate += arrWord[i]
#                     strRemovalDuplicate += ' '
#                 else:
#                     if arrWord[i] == arrWord[i+1]:
#                         strRemovalDuplicate += ' '
#         AnswerWord.append(strRemovalDuplicate)
#     return AnswerWord
#
#
#
# ##匹配抓取结果同义词
# def match(seg):
#     reader = pd.read_csv('synonyms_final.csv')
#     data = pd.DataFrame({'同义词': reader['同义词']})
#     for i in data.index:
#         data1 = data.values[i][0].split(' ')
#         for j in range(1, len(data1)):
#             if data1[j] == seg:
#                 return data.values[i][0]
#     return -1



# ##税收编码表_final同义词分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#
#     print("开始分词",datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#
#     import jieba
#
#     #import jieba_fast as jieba
#     #import jieba_fast.analyse
#
#     import jieba.analyse
#     jieba.load_userdict('ciku.txt')
#     #jieba.enable_parallel(4)#并行分词 仅用于linux系统
#
#     stopwords = [line.rstrip() for line in open('stopword.txt','r', encoding='UTF-8')]
#
#     '''
#     jieba中文分词
#     '''
#     import re
#     #匹配中文的分词
#     #zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     #开始分词，对商品名称进行切割
#     train_X = []
#
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         #segs = jieba.cut(fileContent)#全模式
#         # segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         segs = jieba.cut(fileContent, cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#
#                     Synonyms = match_synonyms('税收编码表_final.txt')
#                     for i in range(len(Synonyms)):
#                         Synonym = Synonyms[i].split(" ")
#                         for j in range(len(Synonym)):
#                             if seg == Synonym[j]:
#                                 Synony = Synonyms[i].split("\n")
#                                 segments.append(Synony[0])
#
#                                 #segments.append(Synonyms[i])
#
#                     segments.append(seg)
#                 num += 1
#         if num>2:
#             fileContent = " ".join(segments)
#         else:
#             fileContent = " ".join(segments+segments)
#         train_X.append(fileContent)
#     return train_X
#
#
#
# ##匹配税收编码表_final同义词
# def match_synonyms(filename):
#     f = open(filename, mode='r', encoding='utf-8')
#     lines = f.readlines()
#     AnswerWord = []
#     for line in lines:
#         strRemovalDuplicate = ''
#         arrWord = line.split(" ")
#         for i in range(len(arrWord)):
#             # print(i, arrWord[i])
#             if i == 0:
#                 strRemovalDuplicate += arrWord[i]
#                 strRemovalDuplicate += ' '
#             else:
#                 if arrWord[i] != arrWord[0]:
#                     strRemovalDuplicate += arrWord[i]
#                     strRemovalDuplicate += ' '
#                 else:
#                     if arrWord[i] == arrWord[i+1]:
#                         strRemovalDuplicate += ' '
#         AnswerWord.append(strRemovalDuplicate)
#     return AnswerWord








# ##匹配词林近义词词典抓取结果分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#     print("开始分词", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#
#     import jieba.analyse
#     # jieba.enable_parallel(4)  # 并行分词 仅用于linux系统
#     # jieba.load_userdict('ciku.txt')
#
#
#     '''
#     jieba中文分词
#     '''
#     import re
#     # 匹配中文的分词
#     # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     # 开始分词，对商品名称进行切割
#     train_X = []
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         # segs = jieba.cut(fileContent)#全模式
#         segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         # segs = jieba.cut(fileContent,cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#                     segments.append(seg)
#                     synonyms = match(seg)
#                     if synonyms != -1:
#                         segments.append(synonyms)
#                 num += 1
#         if num > 2:
#             fileContent = ' '.join(segments)
#         else:
#             fileContent = ' '.join(segments + segments)
#         train_X.append(fileContent)
#     return train_X
#
#
#
# ##匹配抓取结果同义词
# def match(seg):
#     reader = pd.read_csv('synonyms_final.csv')
#     data = pd.DataFrame({'同义词': reader['同义词']})
#     for i in data.index:
#         data1 = data.values[i][0].split(' ')
#         for j in range(1, len(data1)):
#             if data1[j] == seg:
#                 return data.values[i][0]
#     return -1






# ##多线程分词
# def cut_jieba(x):
#
#     import re
#
#     # 匹配中文的分词
#     # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern  = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     outstr = []
#     segs = jieba.cut(x,cut_all=False)
#     for seg in segs:
#         if zhPattern.search(seg):
#             if seg not in stopwords:
#                 outstr.append(seg)
#     if len(outstr)>=2:
#         out = " ".join(outstr)
#     else:
#         out = " ".join(outstr+outstr)
#     return out



# ##知识图谱分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#
#     print("开始分词",datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#     #import jieba_fast as jieba
#     #import jieba_fast.analyse
#
#     import jieba.analyse
#     jieba.load_userdict('ciku.txt')
#     #jieba.enable_parallel(4)#并行分词 仅用于linux系统
#
#     stopwords = [line.rstrip() for line in open('stopword.txt','r', encoding='UTF-8')]
#
#     '''
#     jieba中文分词
#     '''
#     import re
#
#     import json
#     from urllib.parse import quote
#     import string
#     import requests
#
#     #匹配中文的分词
#     #zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     #开始分词，对商品名称进行切割
#     train_X = []
#
#     a = 0
#
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         #segs = jieba.cut(fileContent)#全模式
#         segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         #segs = jieba.cut(fileContent,cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#
#                     segments.append(seg)
#
#                     url = 'http://shuyantech.com/api/cndbpedia/ment2ent?q='+seg+'&apikey=cd9ad381122aeade145b3c55258d9be7'
#                     r = requests.get(quote(url, safe=string.printable))
#
#                     a += 1
#                     print(a)
#
#                     str0 = json.loads(r.text)
#                     rValue = str0['ret']
#
#                     if len(rValue) > 0:
#                         str1 = ''.join(rValue[0])
#                         if str1.find('（') != -1:
#                             str2 = str1.split('（')[1].split('）')[0]
#                             seg = str2
#                         segments.append(seg)
#                     num += 1
#
#
#                 num += 1
#
#
#         if num>2:
#             fileContent = " ".join(segments)
#         else:
#             fileContent = " ".join(segments+segments)
#         train_X.append(fileContent)
#
#     return train_X



# ##税收编码表同义词分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#
#     print("开始分词",datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#
#     import jieba
#
#     #import jieba_fast as jieba
#     #import jieba_fast.analyse
#
#     import jieba.analyse
#     jieba.load_userdict('ciku.txt')
#     #jieba.enable_parallel(4)#并行分词 仅用于linux系统
#
#     stopwords = [line.rstrip() for line in open('stopword.txt','r', encoding='UTF-8')]
#
#     '''
#     jieba中文分词
#     '''
#     import re
#     #匹配中文的分词
#     #zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     #开始分词，对商品名称进行切割
#     train_X = []
#
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         #segs = jieba.cut(fileContent)#全模式
#         segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         #segs = jieba.cut(fileContent,cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#
#                     Synonyms = match_synonyms('synonyms.txt')
#                     for i in range(len(Synonyms)):
#                         Synonym = Synonyms[i].split(" ")
#                         for j in range(len(Synonym)):
#                             if seg == Synonym[j]:
#                                 Synony = Synonyms[i].split("\n")
#                                 segments.append(Synony[0])
#
#                                 #segments.append(Synonyms[i])
#
#                     segments.append(seg)
#                 num += 1
#         if num>2:
#             fileContent = " ".join(segments)
#         else:
#             fileContent = " ".join(segments+segments)
#         train_X.append(fileContent)
#     return train_X
#
#
#
# ##匹配税收编码表同义词
# def match_synonyms(filename):
#     f = open(filename, mode='r', encoding='utf-8')
#     lines = f.readlines()
#     AnswerWord = []
#     for line in lines:
#         strRemovalDuplicate = ''
#         arrWord = line.split(" ")
#         for i in range(len(arrWord)):
#             # print(i, arrWord[i])
#             if i == 0:
#                 strRemovalDuplicate += arrWord[i]
#                 strRemovalDuplicate += ' '
#             else:
#                 if arrWord[i] != arrWord[0]:
#                     strRemovalDuplicate += arrWord[i]
#                     strRemovalDuplicate += ' '
#                 else:
#                     if arrWord[i] == arrWord[i+1]:
#                         strRemovalDuplicate += ' '
#         AnswerWord.append(strRemovalDuplicate)
#     return AnswerWord




# ##词林近义词词典分词
# def sjcl(x):
#     '''
#     jieba分词
#     '''
#     import datetime
#     import time
#     print("开始分词", datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))
#
#     import jieba.analyse
#     # jieba.enable_parallel(4)  # 并行分词 仅用于linux系统
#     # jieba.load_userdict('ciku.txt')
#
#     stopwords = [line.rstrip() for line in open('stopword.txt', 'r', encoding='UTF-8')]
#
#     '''
#     jieba中文分词
#     '''
#     import re
#     # 匹配中文的分词
#     # zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#     zhPattern = re.compile(u'[\u4e00-\u9fa5_a-zA-Z]+')
#
#     # 开始分词，对商品名称进行切割
#     train_X = []
#     for i in range(len(x)):
#         num = 0
#         segments = []
#         fileContent = x[i]
#         #segs = jieba.cut(fileContent)#全模式
#         segs = jieba.cut_for_search(fileContent)#搜索引擎模式
#         #segs = jieba.cut(fileContent,cut_all=False)#精准模式
#         for seg in segs:
#             if zhPattern.search(seg):
#                 if seg not in stopwords:
#                     Synonyms = get_synonyms('seg')
#                     time.sleep(30)
#                     segments.append(Synonyms)
#                     segments.append(seg)
#                 num += 1
#         if num > 2:
#             fileContent = " ".join(segments)
#         else:
#             fileContent = " ".join(segments + segments)
#         train_X.append(fileContent)
#     return train_X



# ##词林近义词词典获取近义词
# def get_synonyms(x):
#
#     ##提取HTML页面中的汉字
#     import requests
#     from bs4 import BeautifulSoup
#     import re
#     zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
#
#     user_agent = 'Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0'
#     headers = {'User-Agent': user_agent}
#     session = requests.session()
#
#     page = session.get('https://www.cilin.org/jyc/w_'+x+'.html', headers=headers)
#
#     # page = requests.get('https://www.cilin.org/jyc/w_'+x+'.html')
#     page.encoding = 'utf-8'
#     soup = BeautifulSoup(page.text, 'lxml')
#     result = ''
#     if soup.title.text !='404错误页面_词林在线词典':
#         taglist = soup.find_all('p', attrs={'class': re.compile("(odd)|()")})
#         for index in range(len(taglist)):
#             WordContent = taglist[index].text
#             if WordContent.find('汉语') != -1:
#                 TargetContent = WordContent.split(":")
#                 Target = TargetContent[1]
#                 match = zhPattern.search(Target)
#                 if match:
#                     result += Target + ', '
#     result = result.replace(',','')
#     result = re.sub('[汉语英语德语法语俄语葡萄牙语西班牙语日语韩语]', '', result)
#     return result






def get_hv(x_train):
    '''
    get HashingVectorizer
    '''
    from sklearn.feature_extraction.text import HashingVectorizer 
    hv = HashingVectorizer(decode_error='ignore', n_features=2 ** 7, norm='l2')
    x_train_hv = hv.transform(x_train)
    return x_train_hv


