import pandas as pd
import numpy as np
import random

from sklearn.model_selection import cross_val_score
train_idx = pd.read_csv('E:\pythonProject\ELMEmotion\data\Group_C/training.csv', index_col=0,encoding='gbk')
test_idx = pd.read_csv('E:\pythonProject\ELMEmotion\data\Group_C/testing.csv', index_col=0,encoding='gbk')

data = [train_idx,test_idx]
data_df = pd.concat(data)
data_df.to_csv("E:\pythonProject\ELMEmotion\data\Group_A/all.csv")

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
regressor = RandomForestClassifier(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)

# Use the trained model to predict the labels for the test set.



f1 = f1_score(y_true = y_test,
              y_pred = predictions)
recall = recall_score(y_true = y_test,
              y_pred = predictions)
accuracy = accuracy_score(y_true = y_test,
              y_pred = predictions)
predictions = precision_score(y_true = y_test,
              y_pred = predictions)



print('Using RandomForest on features...')
print('\nF1: %.3f' % f1)
print ('\nRecall: %.3f' % recall)
print('\nAccuracy: %.3f' % accuracy)
print ('\nPredictions: %.3f' % predictions)