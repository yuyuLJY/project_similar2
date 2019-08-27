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

# TODO 计算type相似度
def count_wv_similar(wv1, wv2):
    '''
    计算矩阵与向量余弦相识度
    :param wv1:
    :param wv2:
    :return:
    '''
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

label = 'test2'
type1_vecs = np.load('wv300_win100.'+label+'_A_Title.npy')
type2_vecs = np.load('wv300_win100.'+label+'_R_Title.npy')
data_frame_type = count_wv_similar(type1_vecs, type2_vecs)
title_similar = data_frame_type
# print(merge[['browsed','type_similar','desire_jd_type_id','jd_sub_type']].head(100))

type1_vecs = np.load('wv300_win100.'+label+'_A_Content.npy')
type2_vecs = np.load('wv300_win100.'+label+'_R_Content.npy')
data_frame_type = count_wv_similar(type1_vecs, type2_vecs)
content_similar = data_frame_type
StackingSubmission = pd.DataFrame(title_similar)
StackingSubmission['content_similar'] = content_similar
print('完成工作类型相似度的匹')

#TODO 求cont的长度
def content_length(a,r):
    return abs(len(a)-len(r))
merge = pd.read_csv('../1-prepare/test_merge.csv', sep=',', low_memory=False)
merge['A_Content'] = merge.A_Content.fillna('U')
merge['R_Content'] = merge.R_Content.fillna('U')
merge['Content_length_gap'] = merge.apply(lambda x: content_length(x['A_Content'],x['R_Content']), axis=1)
#merge['Content_length_gap'] = (merge['Content_length_gap'] - merge['Content_length_gap'].min()) / \
#                              (merge['Content_length_gap'].max() - merge['Content_length_gap'].min())
#用mean归一化
merge['Content_length_gap'] = (merge['Content_length_gap'] - merge['Content_length_gap'].mean()) / merge[
    'Content_length_gap'].std()
StackingSubmission['Content_length_gap'] = merge['Content_length_gap']
StackingSubmission.to_csv('feature_collection_'+label+'.csv', sep=',', header=True, index=False, line_terminator="\n")

