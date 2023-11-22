import numpy as np
import pandas as pd
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import adam_v2
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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

print('Found %s unique tokens.' % len(word_index))
# get only the top frequent words on train

train_x = pad_sequences(train_sequences, maxlen=max_sequence_length)
# get only the top frequent words on test
test_x = pad_sequences(test_sequences, maxlen=max_sequence_length)

print(train_x.shape)
print(test_x.shape)

# 标签向量化
# [0,1]: ham;[1,0]:spam
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
X_train = np.reshape(train_x , (train_x .shape[0], train_x .shape[1], 1))
X_test = np.reshape(test_x, (test_x .shape[0], test_x .shape[1], 1))
print("加载数据完成")
#=============================================================================================
#=============================================================================================

learning_rate = 0.001
training_iters = 20
batch_size = 128
display_step = 10

n_hidden = 128

model = Sequential()
model.add(LSTM(n_hidden,
               batch_input_shape=(None, max_sequence_length, 1),
               unroll=True))

model.add(Dense(2))
model.add(Activation('softmax'))
# plot_model(model, to_file='lstm.png',show_shapes='True')

adam = adam_v2.Adam(lr=learning_rate)
model.summary()
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=training_iters,
          verbose=1,
          validation_data=(X_test, test_y))

scores = model.evaluate(X_test, test_y, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])
