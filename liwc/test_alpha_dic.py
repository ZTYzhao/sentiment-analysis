import os.path
import re
import liwc

from collections import Counter
import pandas as pd
from collections import defaultdict
test_dir = os.path.dirname(__file__)


def test_category_names():
    _, category_names = liwc.load_token_parser(os.path.join(test_dir, "alpha.dic"))
    assert category_names == ["A", "Bravo"]

def tokenize(text):
    # you may want to use a smarter tokenizer
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

def test_parse():
    parse, _ = liwc.load_token_parser(os.path.join(test_dir, "alpha.dic"))
    sentence = "Any alpha a bravo charlie Bravo boy"
    tokens = sentence.split()
    matches = [category for token in tokens for category in parse(token)]
    # matching is case-sensitive, so the only matches are "alpha" (A), "a" (A) and "bravo" (Bravo)
    assert matches == ["A", "A", "Bravo"]

def OutputCSV(result_dics, count_dics, lengths, output_dir, titles):
    if output_dir is "":
        raise ValueError("Output directory not specified")
    if len(titles) != len(count_dics):
        raise ValueError("Invalid number of titles")
    if len(lengths) == 0:
        raise ValueError("No transcripts analyzed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # add total word counts to count_dics
    lengths_for_csv = [{"TOTALWORDS": length} for length in lengths]

    pd.DataFrame(count_dics).T.append(pd.DataFrame(lengths_for_csv).T).reset_index().to_csv(
        output_dir + 'LIWCcounts.csv', header=["Category"] + titles)

    # results dic (words for each category)
    pd.DataFrame(result_dics).T.reset_index().to_csv(
        output_dir + 'LIWCwords.csv', header=["Category"] + titles)

    # relative frequency
    relative_freq_dics = []
    for index, dic in enumerate(count_dics):
        temp_dic = defaultdict(int)
        for category in dic:
            temp_dic[category] = dic[category] / lengths[index]
        relative_freq_dics.append(temp_dic)

    pd.DataFrame(relative_freq_dics).T.reset_index().round(4).to_csv(
        output_dir + 'LIWCrelativefreq.csv', header=["Category"] + titles)

    print("Output saved to " + output_dir)


def get_counts(result_dic):
    counts = {}
    for category in result_dic:
        counts[category] = len(result_dic[category])
    return counts


def analysis_helper(str_in,LIWC_categories):
    results = {}

    for category in LIWC_categories:
        results[category] = []

    gettysburg_tokens = tokenize(str_in.lower())



    for token in gettysburg_tokens:
        print(token)
        for category in parse(token):
            print(category)
            results[category].append(token)
    return results

def Analyze(transcripts_in,category_names):

    if type(transcripts_in) is str:
        transcripts = [transcripts_in]
    else:
        transcripts = transcripts_in

    result_dics=[]
    count_dics=[]
    lengths=[]
    for transcript in transcripts:#逐句子分析
        strs = transcript.split(" ")  # 逐词查
        lengths.append(len(strs))
        result_dics.append(analysis_helper(transcript,category_names))

    for dic in result_dics:
        count_dics.append(get_counts(dic))

    return  result_dics, count_dics,lengths

f = open("E:\pythonProject\Confusion-all\data\liwc-test.csv", encoding="utf-8")

# for i in f.readline():
#     dic = {}
#     i = str(i).replace("\n","")
#     i = i.split(",",2)
#     for j in i:
#         dic[j.split(":",1)[0]] = j.split(":",1)[1]
#         flag.append()
# f.close()
# print(flag)
i=0
key_list=[]
transcript=[]
line=f.readline()
transcripts={}
while line:

    key_list.append("Example"+str(i))

    # transcript = '"'+"Example"+str(i)+'"'+":"+" "+line
    transcript.append(line)
    i = i + 1
    line=f.readline()
for i in range(len(key_list)):
    transcripts[key_list[i]]=transcript[i]



output_dir = "E:\pythonProject\Confusion-all/file/"
str_list = []
for key in transcripts:
    str_list.append(transcripts[key])

result_dics = []


LIWCLocation = "E:\pythonProject\Confusion-all\data\LIWC2015 Dictionary.dic"
parse, category_names = liwc.load_token_parser(LIWCLocation)

result_dics, count_dics, lengths = Analyze(str_list,category_names)

OutputCSV(result_dics, count_dics, lengths, output_dir, list(transcripts.keys()))
print('gg')

# LIWC.print(output_dir, list(transcripts.keys()))



gettysburg = '''Four score and seven years ago our fathers brought forth on
  this continent a new nation, conceived in liberty, and dedicated to the
  proposition that all men are created equal. Now we are engaged in a great
  civil war, testing whether that nation, or any nation so conceived and so
  dedicated, can long endure. We are met on a great battlefield of that war.
  We have come to dedicate a portion of that field, as a final resting place
  for those who here gave their lives that that nation might live. It is
  altogether fitting and proper that we should do this.'''.lower()

gjy_instance = '''This is a single transcript. Red hat angry.'''.lower()



gettysburg_tokens = tokenize(gjy_instance)
# print('ggg')
for token in gettysburg_tokens:
    print(token)
    for category in parse(token):
        print(category)


gettysburg_counts = Counter(category for token in gettysburg_tokens for category in parse(token))
print(gettysburg_counts)
#=> Counter({'funct': 58, 'pronoun': 18, 'cogmech': 17, ...})