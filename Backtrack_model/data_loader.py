from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from tqdm import tqdm
import torch
import pandas as pd
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor


def post_preprocessor():
    preprocessor = TextPreProcessor(
        all_caps_tag="wrap",
        fix_text=False,
        unpack_hashtags=True,
        unpack_contractions=True,
        spell_correct_elong=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
    return preprocessor


class DataClass(Dataset):
    def __init__(self, args, filename):
        self.args = args
        self.filename = filename
        self.max_length = int(args['--max-length'])
        self.data, self.labels = self.load_dataset()

        if args['--lang'] == 'English':
            self.bert_tokeniser = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.inputs, self.lengths, self.label_indices = self.process_data()

    def load_dataset(self):
        #df = pd.read_csv(self.filename, sep='\t')#//当采用train-sentiment.txt时用此注销部分代码。
        #x_train, y_train = df.Tweet.values, df.iloc[:, 2:].values #//当采用train-sentiment.txt时用此注销部分代码。
        df=pd.read_csv(self.filename)
        x_train, y_train = df['text'], df['Confusion']
        return x_train, y_train

    def process_data(self):
        desc = "PreProcessing dataset {}...".format('')
        preprocessor = post_preprocessor()

        if self.args['--lang'] == 'English':
            segment_a = "confusion ?"
            label_names = ["confusion"]

        inputs, lengths, label_indices = [], [], []
        for x in tqdm(self.data, desc=desc):
            x = ' '.join(preprocessor(x))
            x = self.bert_tokeniser.encode_plus(segment_a,
                                                x,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)
            label_idxs = [self.bert_tokeniser.convert_ids_to_tokens(input_id).index(label_names[idx])
                             for idx, _ in enumerate(label_names)]
            label_indices.append(label_idxs)

        inputs = torch.tensor(inputs, dtype=torch.long)
        data_length = torch.tensor(lengths, dtype=torch.long)
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return inputs, data_length, label_indices

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        label_idxs = self.label_indices[index]
        length = self.lengths[index]
        return inputs, labels, length, label_idxs

    def __len__(self):
        return len(self.inputs)
