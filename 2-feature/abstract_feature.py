# -*- coding: utf-8 -*-
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import math
from sklearn import preprocessing
import scipy.stats
from sklearn.preprocessing import StandardScaler
#导入PCA算法库
from sklearn.decomposition import PCA

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 500)

#TODO 查看相似度
'''
data = pd.read_csv('feature_collection.csv', sep=',', low_memory=False)
#data = data[data['delivered']==1]
#data.drop(columns=['area_similar','experience_similar'],inplace=True)
print(data[['area_similar','industry_type_jieba','job_d_jieba']].head(20))
print('*'*60)
print(data[['experience_similar','experience_jieba','power_d_jieba']].head(20))
exit()
'''
#TODO 扔掉特征集合的某些特征
'''
data = pd.read_csv('feature_collection.csv', sep=',', low_memory=False)
#print(data.columns)
print(data.info())
exit()
data.drop(columns=['\\N', '中专', '中技', '初中', '博士', '大专',
       '本科', '硕士', '高中', 'EMBA', 'LOSE', 'MBA', '中专.1', '中技.1', '其他', '初中.1',
       '博士.1', '大专.1', '本科.1', '硕士.1', '高中.1'],inplace=True)
data.to_csv("feature_collection.csv", sep=',', header=True, index=False, line_terminator="\n")
exit()
'''

# TODO (1) 计算type相似度
# 计算矩阵与向量余弦相识度
def count_wv_similar(wv1, wv2):
    list = []
    for i in range(wv1.shape[0]):
        row1 = wv1[i]
        row2 = wv2[i]
        sum = 0
        sq1 = 0
        sq2 = 0
        # print(row1)
        for j in range(len(row1)):
            sum += row1[j] * row2[j]  # 分子
            sq1 += math.pow(row1[j], 2)
            sq2 += math.pow(row2[j], 2)
        # 本行计算完毕
        if sq1 * sq2 == 0:
            result = 0
        else:
            result = float(sum) / (math.sqrt(sq1) * math.sqrt(sq2))
        list.append(result)
    # 全部结果都已经计算完毕
    # print(list)
    data_frame = DataFrame({'type_similar': list})
    return data_frame

type1_vecs = np.load('wv300_win100.test_A_Title.npy')
type2_vecs = np.load('wv300_win100.test_R_Title.npy')
data_frame_type = count_wv_similar(type1_vecs, type2_vecs)
title_similar = data_frame_type
# print(merge[['browsed','type_similar','desire_jd_type_id','jd_sub_type']].head(100))

type1_vecs = np.load('wv300_win100.test_A_Content.npy')
type2_vecs = np.load('wv300_win100.test_R_Content.npy')
data_frame_type = count_wv_similar(type1_vecs, type2_vecs)
content_similar = data_frame_type
print('完成工作类型相似度的匹配')

#'content_similar':np.array(content_similar)}
StackingSubmission = pd.DataFrame(title_similar)
StackingSubmission['content_similar'] = content_similar
StackingSubmission.to_csv('feature_collection_test.csv', sep=',', header=True, index=False, line_terminator="\n")

