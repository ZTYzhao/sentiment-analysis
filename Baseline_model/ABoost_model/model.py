import pandas as pd
import random
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
# First, change the type of the specified columns to "category". This will
# assign a "code" to each unique category value.
#data_df = pd.read_csv('E:\Spanemotion\SpanEmo-master\SpanEmo-LIWC\scripts\confusion-test.csv', index_col=0)
data_df = pd.read_csv(r'E:\pythonProject\Confusion-all\data\confusion-test.csv', index_col=0)
data_df.head()
print(data_df)

# First, calculate the split sizes. 80% training, 10% validation, 10% test.
train_size = int(0.8 * len(data_df))
val_size = int(0.1 * len(data_df))
test_size = len(data_df) - (train_size + val_size)

# Sanity check the sizes.
assert((train_size + val_size + test_size) == len(data_df))

# Create a list of indeces for all of the samples in the dataset.
indeces = np.arange(0, len(data_df))

# Shuffle the indeces randomly.
random.shuffle(indeces)

# Get a list of indeces for each of the splits.
train_idx = indeces[0:train_size]
val_idx = indeces[train_size:(train_size + val_size)]
test_idx = indeces[(train_size + val_size):]

# Sanity check
assert(len(train_idx) == train_size)
assert(len(test_idx) == test_size)

# With these lists, we can now select the corresponding dataframe rows using,
# e.g., train_df = data_df.iloc[train_idx]

print('  Training size: {:,}'.format(train_size))
print('Validation size: {:,}'.format(val_size))
print('      Test size: {:,}'.format(test_size))
# Retrieve the labels for each of the splits.
y_train = data_df["Confusion"].iloc[train_idx]
y_val = data_df["Confusion"].iloc[val_idx]
y_test = data_df["Confusion"].iloc[test_idx]

# Before selecting the inputs, remove text columns and the labels.
# data_df = data_df.drop(columns=["Confusion"])

# Select the inputs for the different splits.
X_train = data_df.iloc[train_idx]
X_val = data_df.iloc[val_idx]
X_test = data_df.iloc[test_idx]

X_train.head()
# Create an instance of the classifier
model = XGBClassifier()

# Train it on the training set.
model.fit(X_train, y_train)

# Use the trained model to predict the labels for the test set.
predictions = model.predict(X_test)

# Calculate the F1 score.
f1 = f1_score(y_true = y_test,
              y_pred = predictions)

print('Using XGBoost on non-text features...')
print('\nF1: %.3f' % f1)