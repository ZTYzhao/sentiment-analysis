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
    print(data_df)


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

        combined += "{:} people read it. ".format(row["reads"])

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
    #
    # print('  DONE.')
    # print('Dataset contains {:,} samples.'.format(len(sen_w_feats)))
    # print(sen_w_feats)
    dict = {"text": sen_w_feats}
    df = pd.DataFrame(dict)
    df.to_csv(r'E:\pythonProject\Confusion-all\Bert-model\post.csv')


DataSet()
