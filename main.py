import nltk
import tqdm  # 进度表
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import csv

# 导入数据
path = './sentiment-analysis-on-movie-reviews/'


# 创建Setence对象，存放每个句子的特征
class Sentence:
    def __init__(self, sentence, sentiment=None):
        self.sentence = sentence
        self.sentiment = sentiment
        self.wordVector = None
        self.sentiVector = [0, 0, 0, 0, 0]


phraseSentimentMap = dict()  # 用于查询每个二元特征（2-gram）对应的情感
sentenceList = []
with open(path + 'train.tsv') as f:
    next(f)
    oldSentenceId = 0
    for content in f:
        content = content.split('\t')
        # 以下四个变量分别为短语和句子id、短语以及短语的情感
        sentenceId, phrase, sentiment = (int(content[1]), content[2], int(content[3]))
        # 采用二元特征（2-gram），建立短语-情感的映射
        if len(phrase.split()) <= 2:
            phrase.strip()
            phraseSentimentMap[phrase] = sentiment
        # 如果是句子，则加入sentenceList
        # 句子一般在同句子序号的第一位
        if sentenceId != oldSentenceId:
            sentenceList.append(Sentence(phrase, sentiment))
            oldSentenceId = sentenceId
# 确定情感矢量
train = []
for s in sentenceList:
    tokens = s.sentence.split()
    tokens2 = list(' '.join(w) for w in nltk.ngrams(tokens, 2))
    vector = np.zeros(5)
    for token in tokens2:
        if token in phraseSentimentMap:
            vector[phraseSentimentMap[token]] += 1
    train.append(vector)
train = np.array(train)

scaler = StandardScaler()  # 构造尺度转化对象scaler
train = scaler.fit_transform(train)  # fit是找出数据的某些特征，transform是基于数据的特征进行转化，注意只有训练集用fit_transform方法
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, class_weight=None)
model2 = copy.deepcopy(model)
trainLabels = np.array([s.sentiment for s in sentenceList])
model.fit(train, trainLabels)


def findSentiment(phraseId):
    # 找到句子的phraseId后返回句子的情感
    # 输入是句子的phraseId
    with open(path + 'sampleSubmission.csv') as f:
        reader = csv.DictReader(f)
        for s in reader:
            # print(s['PhraseId'])
            if int(s['PhraseId'])== phraseId:
                yield s['Sentiment']


testList = []
index = []  # index是句子对应的短语情感
with open(path + 'test.tsv') as f:
    next(f)
    oldSentenceId = 0
    for content in f:
        content = content.split('\t')
        # 以下四个变量分别为短语和句子id、短语
        sentenceId, phrase, phraseId = (int(content[1]), content[2], int(content[0]))
        # 如果是句子，则加入sentenceList
        # 句子一般在同句子序号的第一位
        if sentenceId != oldSentenceId:
            # 获得句子情感
            sentiment = next(findSentiment(phraseId))
            # 将句子加入列表
            testList.append(Sentence(phrase, sentiment))
            # 更新状态变量
            oldSentenceId = sentenceId
