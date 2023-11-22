import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split           #划分训练/测试集
from sklearn.feature_extraction.text import CountVectorizer    #抽取特征
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import nltk
nltk.download('stopwords')
#读取并清洗数据
#因为几个文档的编码不大一样，所以兼容了三种编码模式，根据经验，这三种是经常会遇到的
def get_txt_data(txt_file):
    mostwords=[]
    try:
        file=open(txt_file,'r',encoding='utf-8')
        for line in file.readlines():
            curline=line.strip().split("\t")
            mostwords.append(curline)
    except:
        try:
            file=open(txt_file,'r',encoding='gb2312')
            for line in file.readlines():
                curline=line.strip().split("\t")
                mostwords.append(curline)
        except:
            try:
                file=open(txt_file,'r',encoding='gbk')
                for line in file.readlines():
                    curline=line.strip().split("\t")
                    mostwords.append(curline)
            except:
                ''
    return mostwords
#分词
neg_doc = get_txt_data(r'E:\pythonProject\ELMEmotion\model\Top_Ling_Behav_data\confusion_A.txt')
pos_doc = get_txt_data(r'E:\pythonProject\ELMEmotion\model\Top_Ling_Behav_data\non_confusion_A.txt')


def context_cut(sentence):
    words_list=[]
    words_str=''
    #获取停用词
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    wordsFiltered = []
    for w in words:
        if not(w in stopWords):
            words_list.append(w)
            #wordsFiltered.append(w)
        words_str = ','.join(words_list)
    return words_str,words_list

#合并两个数据集，并且打上标签，分成测试集和训练集
#def Train():

words=[]
word_list=[]
for i in neg_doc:
    cut_words_str,cut_words_list=context_cut(i[0])
    word_list.append((cut_words_str,-1))
    words.append(cut_words_list)
for j in pos_doc:
    cut_words_str2,cut_words_list2=context_cut(j[0])
    word_list.append((cut_words_str2,1))
    words.append(cut_words_list2)
#word_list=[('菜品,质量,好,味道,好,就是,百度,的,问题,总是,用,运力,原因,来,解释,我,也,不,懂,这,是,什么,原因,晚,了,三个,小时,呵呵,厉害,吧,反正,订,了,就,退,不了,只能,干,等', -1),...,...]
#将word_list中的值和标签分别赋予给x,y
x,y=zip(*word_list)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#注意，countvectorizer只能训练str格式的，因为它里面会有lower之类的函数，不适用于列表等其它格式
count = CountVectorizer(max_features=500) #这里的max_features实根据后面的机器学习模型确定的，原先定为200的时候，机器学习效果较差，后改为500效果较好
bag = count.fit_transform(x_train)
    #return x_train, x_test, y_train, y_test
# 引入随机搜索，选择最优模型参数

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train) #与上面一种TfidfTransformer
tfidf_x_test=vec.fit_transform(x_test)

##----------2.AdaBoost-----##
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
AdaBoost = AdaBoostClassifier()
AdaBoost.fit(vec.transform(x_train),y_train)
print ('训练集:',AdaBoost.score(vec.transform(x_train),y_train))  # 精度
print ('训练集:',AdaBoost.score(vec.transform(x_test),y_test))  # 精度

y_train_hat=AdaBoost.predict(vec.transform(x_train))

print ('训练集准确率：',accuracy_score(y_train_hat,y_train))
print ('训练集召回率：',recall_score(y_train_hat,y_train))
print ('F1:',f1_score(y_train_hat,y_train))
