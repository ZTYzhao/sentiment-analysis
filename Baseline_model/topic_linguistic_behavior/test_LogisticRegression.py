import pandas as pd
import numpy as np
import random
train_idx = pd.read_csv('E:\pythonProject\ELMEmotion\data\Group_C/training.csv', index_col=0,encoding='gbk')
test_idx = pd.read_csv('E:\pythonProject\ELMEmotion\data\Group_C/testing.csv', index_col=0,encoding='gbk')

data = [train_idx,test_idx]
data_df = pd.concat(data)
data_df.to_csv("E:\pythonProject\ELMEmotion\data\Group_C/all.csv")

data_df.head()
train_size = int(len(train_idx))
test_size = int(len(test_idx))
indeces = np.arange(0, len(data_df))
# Get a list of indeces for each of the splits.
train_idx = indeces[0:train_size]
val_idx = indeces[train_size:(train_size)]
test_idx = indeces[(train_size):]
assert(len(train_idx) == train_size)
assert(len(test_idx) == test_size)
print('  Training size: {:,}'.format(train_size))
print('      Test size: {:,}'.format(test_size))


# Select the test set samples.
test_df = data_df.iloc[test_idx]

# Create a list of all 1s to use as our predictions.


data_df["content-related"] = data_df["content-related"].astype('float')
data_df["non-content-related"] = data_df["non-content-related"].astype('float')

data_df["up_count"] = data_df["up_count"].astype('float')
data_df["reads"] = data_df["reads"].astype('float')
data_df["Comment"] = data_df["Comment"].astype('float')
data_df["CommentThread"] = data_df["CommentThread"].astype('float')


data_df["function"] = data_df["function"].astype('float')
data_df["reads"] = data_df["reads"].astype('float')
data_df["pronoun"] = data_df["pronoun"].astype('float')
data_df["ppron"] = data_df["ppron"].astype('float')
data_df["i"] = data_df["i"].astype('float')
data_df["we"] = data_df["we"].astype('float')
data_df["you"] = data_df["you"].astype('float')
data_df["reads"] = data_df["reads"].astype('float')
data_df["shehe"] = data_df["shehe"].astype('float')
data_df["they"] = data_df["they"].astype('float')
data_df["ipron"] = data_df["ipron"].astype('float')
data_df["article"] = data_df["article"].astype('float')
data_df["prep"] = data_df["prep"].astype('float')
data_df["auxverb"] = data_df["auxverb"].astype('float')
data_df["adverb"] = data_df["adverb"].astype('float')
data_df["conj"] = data_df["conj"].astype('float')
data_df["negate"] = data_df["negate"].astype('float')
data_df["verb"] = data_df["verb"].astype('float')
data_df["adj"] = data_df["adj"].astype('float')
data_df["compare"] = data_df["compare"].astype('float')
data_df["interrog"] = data_df["interrog"].astype('float')
data_df["number"] = data_df["number"].astype('float')
data_df["quant"] = data_df["quant"].astype('float')
data_df["affect"] = data_df["affect"].astype('float')
data_df["posemo"] = data_df["posemo"].astype('float')
data_df["negemo"] = data_df["negemo"].astype('float')
data_df["anx"] = data_df["anx"].astype('float')
data_df["anger"] = data_df["anger"].astype('float')
data_df["sad"] = data_df["sad"].astype('float')
data_df["social"] = data_df["social"].astype('float')
data_df["family"] = data_df["family"].astype('float')
data_df["friend"] = data_df["friend"].astype('float')
data_df["female"] = data_df["female"].astype('float')
data_df["male"] = data_df["male"].astype('float')
data_df["cogproc"] = data_df["cogproc"].astype('float')
data_df["insight"] = data_df["insight"].astype('float')
data_df["cause"] = data_df["cause"].astype('float')
data_df["discrep"] = data_df["discrep"].astype('float')
data_df["tentat"] = data_df["tentat"].astype('float')
data_df["certain"] = data_df["certain"].astype('float')
data_df["differ"] = data_df["differ"].astype('float')
data_df["percept"] = data_df["percept"].astype('float')
data_df["see"] = data_df["see"].astype('float')
data_df["hear"] = data_df["hear"].astype('float')
data_df["feel"] = data_df["feel"].astype('float')
data_df["bio"] = data_df["bio"].astype('float')
data_df["body"] = data_df["body"].astype('float')
data_df["health"] = data_df["health"].astype('float')
data_df["sexual"] = data_df["sexual"].astype('float')
data_df["ingest"] = data_df["ingest"].astype('float')
data_df["drives"] = data_df["drives"].astype('float')
data_df["affiliation"] = data_df["affiliation"].astype('float')
data_df["achiev"] = data_df["achiev"].astype('float')
data_df["power"] = data_df["power"].astype('float')
data_df["reward"] = data_df["reward"].astype('float')
data_df["risk"] = data_df["risk"].astype('float')
data_df["focuspast"] = data_df["focuspast"].astype('float')
data_df["focuspresent"] = data_df["focuspresent"].astype('float')
data_df["focusfuture"] = data_df["focusfuture"].astype('float')
data_df["relativ"] = data_df["relativ"].astype('float')
data_df["motion"] = data_df["motion"].astype('float')
data_df["space"] = data_df["space"].astype('float')
data_df["time"] = data_df["time"].astype('float')
data_df["work"] = data_df["work"].astype('float')
data_df["leisure"] = data_df["leisure"].astype('float')
data_df["home"] = data_df["home"].astype('float')
data_df["money"] = data_df["money"].astype('float')
data_df["relig"] = data_df["relig"].astype('float')
data_df["death"] = data_df["death"].astype('float')
data_df["informal"] = data_df["informal"].astype('float')
data_df["netspeak"] = data_df["netspeak"].astype('float')
data_df["assent"] = data_df["assent"].astype('float')
data_df["nonflu"] = data_df["nonflu"].astype('float')
data_df["filler"] = data_df["filler"].astype('float')
data_df["confusion expressions"] = data_df["confusion expressions"].astype('float')
data_df["incomplete expressions"] = data_df["incomplete expressions"].astype('float')
data_df["opinion"] = data_df["opinion"].astype('float')
data_df["Pedagogical"] = data_df["Pedagogical"].astype('float')
data_df["future words"] = data_df["future words"].astype('float')


data_df.head()
# Retrieve the labels for each of the splits.
y_train = data_df["Confusion"].iloc[train_idx]
y_test = data_df["Confusion"].iloc[test_idx]

# Before selecting the inputs, remove text columns and the labels.
data_df = data_df.drop(columns=[ "Confusion"])

# Select the inputs for the different splits.
X_train = data_df.iloc[train_idx]
X_test = data_df.iloc[test_idx]

X_train.head()

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
model = LogisticRegression()
# Train it on the training set.
model.fit(X_train, y_train)

# Use the trained model to predict the labels for the test set.
predictions = model.predict(X_test)


f1 = f1_score(y_true = y_test,
              y_pred = predictions)
recall = recall_score(y_true = y_test,
              y_pred = predictions)
accuracy = accuracy_score(y_true = y_test,
              y_pred = predictions)
predictions = precision_score(y_true = y_test,
              y_pred = predictions)



print('Using LogisticRegression on features...')
print('\nF1: %.3f' % f1)
print ('\nRecall: %.3f' % recall)
print('\nAccuracy: %.3f' % accuracy)
print ('\nPredictions: %.3f' % predictions)
