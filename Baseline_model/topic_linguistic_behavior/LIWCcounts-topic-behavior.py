import pandas as pd
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb
# 读取数据
# pandas读入
data = pd.read_csv('/model/topic_linguistic_behavior/LIWCcounts-topic-behavior.csv')  # TV、Radio、Newspaper、Sales
x = data[['function','pronoun','ppron','i','we','you',
          'shehe','they','ipron','article','prep','auxverb','adverb','conj','negate',
          'verb','adj','compare','interrog','number','quant','affect','posemo','negemo','anx',
          'anger','sad','social','family','friend','female','male','cogproc','insight','cause',
          'discrep','tentat','certain','differ','percept','see','hear','feel','bio','body','health',
          'sexual','ingest','drives','affiliation','achiev','power','reward','risk','focuspast','focuspresent',
          'focusfuture','relativ','motion','space','time','work','leisure','home','money','relig',
          'death','informal','swear','netspeak','assent','nonflu','filler','confusion expressions',
          'incomplete expressions','opinion','Pedagogical','future words','content', 'non-content',
          'up_count','reads','Comment','CommentThread',]]

y = data['Confusion']
print(x)
print(y)



lr = LogisticRegression(random_state=2018,tol=1e-6)  # 逻辑回归模型

tree = DecisionTreeClassifier(random_state=2018) #决策树模型

svm = SVC(probability=True,random_state=2018,tol=1e-6)  # SVM模型

forest=RandomForestRegressor(n_estimators=100,random_state=2018) #　随机森林

Ada=AdaBoostClassifier(random_state=2018)  #Ada




def muti_score(model):
    warnings.filterwarnings('ignore')
    accuracy = cross_val_score(model,x, y, scoring='accuracy', cv=10)
    precision = cross_val_score(model,x, y, scoring='precision', cv=10)
    recall = cross_val_score(model ,x, y, scoring='recall', cv=10)
    f1_score = cross_val_score(model ,x, y, scoring='f1', cv=10)
    auc = cross_val_score(model,x, y, scoring='roc_auc', cv=10)
    print("准确率:",accuracy.mean())
    print("精确率:",precision.mean())
    print("召回率:",recall.mean())
    print("F1_score:",f1_score.mean())
    print("AUC:",auc.mean())



model_name=["lr","tree","svm","forest","Ada"]
for name in model_name:
    model=eval(name)
    print(name)
    muti_score(model)


'''
lr
准确率: 0.7890191148682617
精确率: 0.6542724662896913
召回率: 0.3377975457965613
F1_score: 0.44525012166067884
AUC: 0.7840451024530857
tree
准确率: 0.6962524533638791
精确率: 0.39920670173446693
召回率: 0.4157413593052284
F1_score: 0.40705496051057793
AUC: 0.6029856787858856
svm
准确率: 0.787758390223099
精确率: 0.7351623295760905
召回率: 0.24060335431243626
F1_score: 0.36179547264664874
AUC: 0.7640376541388867
forest
准确率: 0.7921756804332226
精确率: 0.7135700690071172
召回率: 0.2867128441334693
F1_score: 0.40835414886475174
AUC: 0.7752164698827589
Gbdt
准确率: 0.7938590063951863
精确率: 0.6604108594441386
召回率: 0.36633732991104395
F1_score: 0.4708811551285791
AUC: 0.7888240065764295
Ada
准确率: 0.7982740847293591
精确率: 0.6829783239831001
召回率: 0.3663162336064133
F1_score: 0.47673826685376613
AUC: 0.7914190511145234
gbm
准确率: 0.79049080811139
精确率: 0.6421783397519263
召回率: 0.3730354066312717
F1_score: 0.47150438344663004
AUC: 0.7776116341798183
'''
