import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Dropout,BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from livelossplot import PlotLossesKeras
import re
import os

os.environ["PATH"] += os.pathsep +'E:\pythonProject\Confusion-all\Baseline_model\models'
# 读取数据
data = pd.read_csv('baseline-test.csv', encoding = "utf-8")
# # 去除无用数据，后3列是无用数据
# data = data[['v1', 'v2']]
# # 修改表头信息
data = data.rename(columns={"v1":"label","v2":"text"})
# 去除标点符号及两个以上的空格
data['text'] = data['text'].apply(lambda x:re.sub('[!@#$:).;,?&]', ' ', x.lower()))
data['text'] = data['text'].apply(lambda x:re.sub(' ', ' ', x))
# 单词转换为小写
data['text'] = data['text'].apply(lambda x:" ".join(x.lower() for x in x.split()))
# 去除停止词 ，如a、an、the、高频介词、连词、代词等
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# 分词处理，希望能够实现还原英文单词原型
st = PorterStemmer()
data['text'] = data['text'].apply(lambda x: " ".join([word for word in x.split()]))
#data['text'] = data['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#分出训练集和测试集
train=data[:5000]
test=data[5000:]
# 每个序列的最大长度，多了截断，少了补0
max_sequence_length = 300

#只保留频率最高的前20000个词
num_words = 20000

# 嵌入的维度
embedding_dim = 100
# 找出经常出现的单词，分词器
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train.text)
train_sequences = tokenizer.texts_to_sequences(train.text)
test_sequences = tokenizer.texts_to_sequences(test.text)

# dictionary containing words and their index
word_index = tokenizer.word_index

# print(tokenizer.word_index)
# total words in the corpus
print('Found %s unique tokens.' % len(word_index))
# get only the top frequent words on train

train_x = pad_sequences(train_sequences, maxlen=max_sequence_length)
# get only the top frequent words on test
test_x = pad_sequences(test_sequences, maxlen=max_sequence_length)
print(train_x.shape)
print(test_x.shape)

def lable_vectorize(labels):
    label_vec = np.zeros([len(labels), 2])
    for i, label in enumerate(labels):
        if str(label) == 'ham':
            label_vec[i][0] = 1
        else:
            label_vec[i][1] = 1
    return label_vec

train_y = lable_vectorize(train['label'])
test_y = lable_vectorize(test['label'])
'''
num_words = 20000
embedding_dim = 100
'''
model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    input_length=max_sequence_length))
#model.add(Embedding(max_sequence_length,128))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))

model.add(BatchNormalization())
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
plot_model(model, to_file='model3.png',show_shapes='True')
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(train_x, train_y,
            batch_size=64,
            epochs=20,
            validation_split=0.2,
          callbacks=[PlotLossesKeras()])
print(model.evaluate(test_x, test_y))
# prediction on test data
predicted=model.predict(test_x)
print(predicted)

#模型评估
import sklearn
from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(test_y,predicted.round())
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print("############################")
print(sklearn.metrics.classification_report(test_y,predicted.round()))
