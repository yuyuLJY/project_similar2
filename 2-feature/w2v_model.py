# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import csv
from gensim.models import word2vec
from gensim import matutils
import scipy
import math
from gensim import corpora, models
from sklearn.decomposition import PCA
from typing import List


# 计算矩阵与向量余弦相识度
def cosine_Matrix(_matrixA, vecB):
    _matrixA_matrixB = np.dot(_matrixA, vecB.T).T
    _matrixA_norm = np.sqrt(np.multiply(_matrixA, _matrixA).sum(axis=1))
    vecB_norm = np.linalg.norm(vecB)
    print('计算矩阵与向量余弦相识度')
    return np.divide(_matrixA_matrixB, _matrixA_norm * vecB_norm.transpose())

def create_industry_similar(describe_wv,industry_wv):
    #计算相似度，并返回
    cosresult = cosine_Matrix(describe_wv, industry_wv)  # 调用函数
    cosresult = cosresult.tolist()
    # sort_cosresult = sorted(cosresult)  # 从小到大
    # print(sort_cosresult)
    print('排序完毕')

    # 找出跟句子1，最相近的其他句子
    # 打印出最相近的，相似度最大的前8个
    print(cosresult[0])  # 相似度
    #print(traindatalist[76])  # 找到那行，idx是原来训练数据的行
    return cosresult[0]

def train_w2v(size, filename):
    """
    训练wv模型
    :param filename:path
    :return:none
    """
    sentences = word2vec.LineSentence(filename)  # 加载语料，要求语料为“一行一文本”的格式
    print('正在训练w2v 针对语料：', str(filename))
    print('size is: ', size)
    model = word2vec.Word2Vec(sentences, size=size, window=100, workers=48)  # 训练模型; 注意参数window 对结果有影响 一般5-100
    savepath = '1size_win100_' + str(size) + '.model'  # 保存model的路径
    print('训练完毕，已保存: ', savepath, )
    model.save(savepath)

# ==============词向量求平均===================
def sentenceByWordVectAvg(sentenceList, model, embeddingSize):
    sentenceSet = []
    for sentence in sentenceList:
        # 将所有词向量的woed2vec向量相加到句向量
        sentenceVector = np.zeros(embeddingSize)
        # 计算每个词向量的权重，并将词向量加到句向量
        row_word = sentence.split(' ')
        count = 0
        for word in row_word:
            try:
                count+=1
                sentenceVector = np.add(sentenceVector, model[word])
            except:
                #print('没有这个词'+word)
                pass
        if count!=0:
            sentenceVector = np.divide(sentenceVector, float(count))
        sentenceSet.append(sentenceVector)
    print('求平均词向量完毕')
    return sentenceSet

def gettrainDatalist(data_jieba):
    train_data = np.array(data_jieba)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    #TODO 新添加的，需要修改ave_w2v函数
    train_x_list = list(map(lambda x: x.split(' '), train_x_list))
    return train_x_list

def saveIndex(sentence_vecs):
    # TODO 这个index是什么意思？？？
    corpus_len = len(sentence_vecs)
    print(corpus_len)
    index = np.empty(shape=(corpus_len, 300), dtype=np.float32)
    for docno, vector in enumerate(sentence_vecs):
        if isinstance(vector, np.ndarray):
            pass
        elif scipy.sparse.issparse(vector):
            vector = vector.toarray().flatten()
        else:
            vector = matutils.unitvec(matutils.sparse2full(vector, 300))
        index[docno] = vector
    return index

def ave_w2v(X, model, size):
    """
    方法1：计算向量的平均值
    载入模型，并且生成wv向量
    :param X:读入的文档，list
    :return:np.array
    """
    # 返回来一个给定形状和类型的用0填充的数组
    res = np.zeros((len(X), size))
    print('生成w2v向量中..')
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for i, line in enumerate(X):
        terms = str(line).split()
        count = 0
        for j, term in enumerate(terms):
            try:  # ---try失败说明X中有单词不在model中，训练的时候model的模型是min_count的 忽略了一部分单词
                count += 1
                res[i] += np.array(model[term])
            except:
                1 == 1
        if count != 0:
            res[i] = res[i] / float(count)  # 求均值
    return res

def list_dict(list_data):
    list_data = list(map(lambda x: {str(x[0]): x[1]}, list_data))
    dict_data = {}
    for i in list_data:
        key, = i
        value, = i.values()
        dict_data[key] = value
    return dict_data

def ave_ifidf_w2v(sentenceList, model, size):
    '''
    方法2：加入if-idf权重
    :param sentenceList:
    :param model:
    :param size:
    :return:
    '''
    dictionary = corpora.Dictionary(sentenceList)  ##得到词典
    token2id = dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in sentenceList]  ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    print('dictionary prepared!')
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    corpus_tfidf = tfidf[corpus]

    sentenceSet = []
    for i in range(len(sentenceList)):
        # 将所有词向量的woed2vec向量相加到句向量
        sentenceVector = np.zeros(size)
        # 计算每个词向量的权重，并将词向量加到句向量
        sentence = sentenceList[i]
        sentence_tfidf = corpus_tfidf[i]
        dict_tfidf = list_dict(sentence_tfidf)
        for word in sentence:
            try:
                tifidf_weigth = dict_tfidf.get(str(token2id[word]))
                sentenceVector = np.add(sentenceVector, tifidf_weigth * model[word])
            except:
                pass
        sentenceVector = np.divide(sentenceVector, len(sentence))
        # 存储句向量
        sentenceSet.append(sentenceVector)
    return np.array(sentenceSet)

# ===============sentence2vec：词向量加权-PCA==================
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector

    # a sentence, a list of words


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

        # return the length of a sentence

    def len(self) -> int:
        return len(self.word_list)

    # convert a list of sentence with word2vec items into a set of sentence vectors

def w2v_pca(traindata,sentenceList: List[Sentence],embeddingSize,a: float = 1e-3):
    '''
    方法3：词向量加权-PCA
    :param wdfs:
    :param token2id:
    :param sentenceList:
    :param embeddingSize:
    :param charLen:
    :param a:
    :return:
    '''
    dictionary = corpora.Dictionary(traindata)  ##得到词典
    token2id = dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in traindata]  ##统计每篇文章中每个词出现的次数:[(词编号id,次数number)]
    tfidf = models.TfidfModel(corpus=corpus, dictionary=dictionary)
    wdfs = tfidf.dfs

    charLen = dictionary.num_pos
    sentenceSet = []
    for sentence in sentenceList:
        sentenceVector = np.zeros(embeddingSize)
        for word in sentence.word_list:
            p = wdfs[token2id[word.text]] / charLen
            a = a / (a + p)
            sentenceVector = np.add(sentenceVector, np.multiply(a, word.vector))
        if sentence.len()==0:
            sentenceVector = np.zeros(embeddingSize)
        else:
            sentenceVector = np.divide(sentenceVector, sentence.len())
        sentenceSet.append(sentenceVector)
        # caculate the PCA of sentenceSet
    pca = PCA(n_components=embeddingSize)
    pca.fit(np.array(sentenceSet))
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))

    # occurs if we have less sentences than embeddings_size
    if len(u) < embeddingSize:
        for i in range(embeddingSize - len(u)):
            u = np.append(u, [0])

            # remove the projections of the average vectors on their first principal component
    # (“common component removal”).
    sentenceVectors = []
    for sentenceVector in sentenceSet:
        sentenceVectors.append(np.subtract(sentenceVector, np.multiply(u, sentenceVector)))
    return np.array(sentenceVectors)

def get_wv(file_name,colunm,model,size,write_to_file,w_to_name,methon):
    merge = pd.read_csv(file_name, sep=',', low_memory=False)
    type1_word = merge[colunm]

    type1_list = gettrainDatalist(type1_word)

    #选择哪个方法：ave_w2v ave_ifidf_w2v
    if methon=='ave_ifidf_w2v':
        type1_vecs = ave_ifidf_w2v(type1_list, model, size)
    if methon=='w2v_pca':
        Sentence_list = []
        for td in type1_list:
            vecs = []
            for s in td:
                try:
                    w = Word(s, model[s])
                    vecs.append(w)
                except:
                    pass
            if len(td) == 0:
                vecs.append(0)
            sentence = Sentence(vecs)
            Sentence_list.append(sentence)
        type1_vecs = w2v_pca(type1_list,Sentence_list,size)

    print(type1_vecs.shape)
    np.save('wv300_win100.'+write_to_file+'_'+w_to_name+'.npy', type1_vecs)

#TODO step1：训练模型
size = 300
#train_w2v(size,'../1-prepare/train_jieba_text.csv') #训练模型
model = word2vec.Word2Vec.load('1size_win100_300.model')

#TODO 提取各个列样本的维度数据
#训练集
'''
write_to_file_list = ['jieba_A_Title.csv', 'jieba_R_Title.csv', 'jieba_A_Content.csv', 'jieba_R_Content.csv']
colunm_list = ['A_Title', 'R_Title', 'A_Content', 'R_Content']
for i, j in zip(write_to_file_list, colunm_list):
    #ave_ifidf_w2v w2v_pca
    get_wv('../1-prepare/'+i,j,model,size,'train2',j,'ave_ifidf_w2v')
'''

#测试集
write_to_file_list = ['test_jieba_A_Title.csv', 'test_jieba_R_Title.csv', 'test_jieba_A_Content.csv', 'test_jieba_R_Content.csv']
colunm_list = ['A_Title', 'R_Title', 'A_Content', 'R_Content']
for i, j in zip(write_to_file_list, colunm_list):
    get_wv('../1-prepare/' + i, j, model, size, 'test2', j, 'ave_ifidf_w2v')

