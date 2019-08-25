# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import csv
from gensim.models import word2vec
from gensim import matutils
import scipy
import math

#TODO ：训练一个model
#TODO 目的：创建两行新的列,对于每一个jb_no,都有一个对应的向量
#TODO 对于每一个require_industry，也有一个向量

#TODO 现在多了两列 describe_wv,industry_wv,由这两行产生新的一列：match_industry

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

# 获取训练数据
def gettrainDatalist(trainname):
    """
    将每一行的句子变成list
    :param trainname:path
    :return: list
    """
    traindata = []
    with open(trainname, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        count = 0
        for line in reader:
            try:
                traindata.append(line[0])
                count += 1
            except:
                print("error:", line, count,line)
                traindata.append(" ")
    print('读取数据完毕')
    return traindata

def gettrainDatalist1(data_jieba):
    train_data = np.array(data_jieba)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
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


def get_require_industry_wv(require_industry,model,size):
    #require_industry:"工程造价 预结算"
    #querylist = ["工程造价 预结算"]
    if str(require_industry).__eq__('N'):
        return 'N'  #期望的领域写“其他”，填上匹配度0.5
    else:
        querylist = []
        querylist.append(require_industry)
        query_wv = sentenceByWordVectAvg(querylist, model, size)[0]
        #print(query_wv)
        s = ""
        for i in query_wv:
            # row是一个列表
            s = s + str(i) + " "
        s = s.strip()
        return s

def get_list_vm(vm):
    list = []
    list.append(vm)
    return list

def write_industry_wv_tocsv(path, head, data):
    try:
        with open(path, 'w', newline='',encoding='utf8') as csv_file:
            writer = csv.writer(csv_file)
            if head is not None:
                writer.writerow(head)
            count=-1
            for row in data:
                #row是一个列表
                count+=1
                print(count)
                s =""
                if float(row[0])==0 and float(row[1])==0:
                    writer.writerow('N')
                else:
                    for i in row:
                        s = s + str(i) + " "
                    s = s.strip()
                    # writer.writerow('N')
                    writer.writerow([s])
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))

def load_trainsform(X, model, size):
    """
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

def get_wv(file_name,colunm,model,size,write_to_file,w_to_name):
    merge = pd.read_csv(file_name, sep=',', low_memory=False)
    type1_word = merge[colunm]

    # 三个list的长度都是：648171
    type1_list = gettrainDatalist1(type1_word)

    type1_vecs = load_trainsform(type1_list, model, size)

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
    get_wv('../1-prepare/'+i,j,model,size,'train',j)
'''

#测试集
write_to_file_list = ['test_jieba_A_Title.csv', 'test_jieba_R_Title.csv', 'test_jieba_A_Content.csv', 'test_jieba_R_Content.csv']
colunm_list = ['A_Title', 'R_Title', 'A_Content', 'R_Content']
for i, j in zip(write_to_file_list, colunm_list):
    get_wv('../1-prepare/'+i,j,model,size,'test',j)

