import pandas as pd
import random
# import numpy as np
# import torch
# from sklearn.metrics import f1_score
# from transformers import BertTokenizer
# from transformers import BertForSequenceClassification
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from transformers import AdamW
# from transformers import get_linear_schedule_with_warmup
import numpy as np
# import time
# import datetime
# from sklearn.metrics import f1_score
def DataSet():
    #data_df = pd.read_csv('E:\Spanemotion\SpanEmo-master\SpanEmo-LIWC\scripts\confusion-test.csv', index_col=0)
    data_df = pd.read_csv(r'E:\pythonProject\Confusion-all\data\confusion-test.csv', index_col=0)
    data_df.head()
    # print(data_df)
    # First, calculate the split sizes. 80% training, 10% validation, 10% test.训练-验证-测试拆分
    train_size = int(0.8 * len(data_df))
    val_size = int(0.1 * len(data_df))
    test_size = len(data_df) - (train_size + val_size)

    # Sanity check the sizes.
    assert ((train_size + val_size + test_size) == len(data_df))

    # Create a list of indeces for all of the samples in the dataset.
    indeces = np.arange(0, len(data_df))

    # Shuffle the indeces randomly.
    random.shuffle(indeces)

    # Get a list of indeces for each of the splits.
    train_idx = indeces[0:train_size]
    val_idx = indeces[train_size:(train_size + val_size)]
    test_idx = indeces[(train_size + val_size):]

    # Sanity check
    assert (len(train_idx) == train_size)
    assert (len(test_idx) == test_size)

    # With these lists, we can now select the corresponding dataframe rows using,
    # e.g., train_df = data_df.iloc[train_idx]

    print('  Training size: {:,}'.format(train_size))
    print('Validation size: {:,}'.format(val_size))
    print('      Test size: {:,}'.format(test_size))

    # This will hold all of the dataset samples, as strings.

    sen_w_feats = []

    # The labels for the samples.
    labels = []
    for index, row in data_df.iterrows():
        # Piece it together...
        combined = ""

        # combined += "The ID of this item is {:}, ".format(row["Clothing ID"])
        # combined += "This item comes from the {:} department and {:} division, " \
        #             "and is classified under {:}. ".format(row["Department Name"],
        #                                                    row["Division Name"],
        #                                                    row["Class Name"])


        if row["reads"]>0:
            combined += "{:} people read it. ".format(row["reads"])
        if row["up_count"]>0:
            combined += "{:} people up_count it. ".format(row["up_count"])
        combined = combined + index
        LIWC_feature= list(row[5:][row[5:] > 0].index)
        combined = combined + "The LIWC features are {:}".format(' and '.join(LIWC_feature))

        # # Not all samples have titles.
        # if not row["Title"] == "":
        #     combined += row["Title"] + ". "

        # Finally, append the review the text!
        # combined += row["Text"]

        # Add the combined text to the list.
        sen_w_feats.append(combined)

        # Also record the sample's label.
        labels.append(int(row["Confusion"]))

    print('  DONE.')
    print('Dataset contains {:,} samples.'.format(len(sen_w_feats)))
    return  sen_w_feats, labels,train_idx,val_idx,test_idx


