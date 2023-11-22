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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count = CountVectorizer(max_features=500) #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
tfidf = TfidfTransformer(use_idf = True, norm = 'l2', smooth_idf = True) #该类会统计每个词语的tf-idf权值
tfidf_x_train=tfidf.fit_transform(count.fit_transform(x_train)) #将文本转为词频矩阵并计算tf-idf
np.set_printoptions(precision = 2)
x_train_weight = tfidf_x_train.toarray()
#还有另一种方法是TfidfVectorizer，
#关于TfidfVectorizer和TfidfTransformer的关联可以参照[TfidfVectorizer和TfidfTransformer](https://blog.csdn.net/liuchenbaidu/article/details/105063535?biz_id=102&utm_term=TfidfVectorizer&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-105063535&spm=1018.2118.3001.4187)
from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_features=500)
tfidf_x_train=vec.fit_transform(x_train) #与上面一种TfidfTransformer
tfidf_x_test=vec.fit_transform(x_test)



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
forest_reg=RandomForestRegressor()
forest_reg.fit(vec.transform(x_train),y_train)
print('训练集:',forest_reg.score(vec.transform(x_test),y_test))
print('测试集:',forest_reg.score(vec.transform(x_train),y_train))
'''训练集: 0.5592796842264617
测试集: 0.8471523627994759'''

##引入交叉验证，对模型进行调优
x_train_tf=tfidf.fit_transform(count.fit_transform(x_train)).toarray()
forest_scores=cross_val_score(forest_reg,x_train_tf,y_train,scoring='neg_mean_squared_error',cv=10)



# print ('训练集:',forest_reg.score(vec.transform(x_train), y_train))  # 精度
# print ('测试集:',forest_reg.score(vec.transform(x_test), y_test))

y_train_hat=forest_reg.predict(vec.transform(x_train))
print ('训练集准确率：',accuracy_score(y_train_hat,y_train))
print ('训练集召回率：',recall_score(y_train_hat,y_train))
print ('F1:',f1_score(y_train_hat,y_train))
print ('ROC值：',roc_auc_score(y_train_hat,y_train))


# 分类报告：precision/recall/fi-score/均值/分类个数
from sklearn.metrics import classification_report
target_names = ['-1','1']
print(classification_report(y_train_hat,y_train, target_names=target_names))

