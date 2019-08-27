# coding=utf-8
import classify
import pandas as pd
import numpy as np
import csv
import codecs
import multiprocessing
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#导入PCA算法库
from sklearn.decomposition import PCA
import warnings


warnings.filterwarnings("ignore")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def gettrainDatalist1(data_jieba):
    train_data = np.array(data_jieba)  # np.ndarray()
    train_x_list = train_data.tolist()  # list
    return train_x_list


def output(filename, ID, age, gender, education):
    """
    generate the submit file
    :param filename: path of the submit file
    :param ID: user ID
    :param age:predicted age
    :param gender:predicted gender
    :param education:predicted education
    :return:submit file
    """
    print(ID.shape, age.shape, gender.shape, education.shape)
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        count = 0
        for i in range(len(ID)):
            # if count>=1000:
            #     break
            f.write(str(ID[i]) + ' ' + str(age[i]) + ' ' + str(gender[i]) + ' ' + str(education[i]) + '\n')
            count += 1

def reduce_pca(data,name,rate):
    '''
    传入：dataframe,返回一个dataframe
    :return:
    '''
    data = np.array(data)
    sc = StandardScaler()
    data = sc.fit_transform(data)

    # 降维 创建PCA对象，n_components用float表示，保留多少的特征n_components=0.9
    pca = PCA(n_components=rate)
    # 使用PCA对特征进行降维
    data_std = pca.fit_transform(data)

    column_list = []
    for i in range(data_std.shape[1]):
        column_list.append(name+str(i))
    data_std = pd.DataFrame(data_std)
    data_std.columns = column_list
    return data_std

if __name__ == '__main__':
    """
    the main function
    注意路径
    """
    start = time.time()
    order = 'predict'  # execute predict function
    #order='test' #execute 2-fold validation function
    #order = 'combine'
    #print('orderis ', order)
    print('----------start----------')

    if order == 'test':
        # TODO 读取标签
        train_merge = pd.read_csv('../1-prepare/train_merge.csv', sep=',', low_memory=False)
        #train_feature = train_feature.iloc[0:10000, :]
        #print(train_feature.columns)
        #print('训练数据读取完毕')

        #TODO 相似性
        f1 = pd.read_csv('../2-feature/feature_collection_train2.csv', sep=',', low_memory=False)

        #TODO 读取特征
        type1_vecs = np.load('../2-feature/wv300_win100.train_A_Title.npy') #10W行
        type2_vecs = np.load('../2-feature/wv300_win100.train_R_Title.npy')
        type3_vecs = np.load('../2-feature/wv300_win100.train_A_Content.npy') #10W行
        type4_vecs = np.load('../2-feature/wv300_win100.train_R_Content.npy')

        #变成
        feature = np.concatenate((type1_vecs,type2_vecs,type3_vecs,type4_vecs),axis=1)
        feature = reduce_pca(feature, 'c',0.37)
        print('完成降维处理')
        train_feature = pd.DataFrame(feature)
        #f1[['type_similar','content_similar']]
        train_feature = pd.concat([train_feature,f1], axis=1)
        print('所有特征')
        print(train_feature.info())
        # 进行三折运算
        kf = KFold(n_splits=3, shuffle=False, random_state=1)
        predictions = []
        for train, test in kf.split(train_feature):
            # The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
            train_predictors = (train_feature.iloc[train, :])
            # The target we're using to train the algorithm.
            train_target = train_merge['Level'].iloc[train]
            test_predictions = classify.term().predict(train_predictors, train_target,
                                                       train_feature.iloc[test, :], 'gender')
            predictions.append(test_predictions)

        # 将结果写入csv
        predictions = np.concatenate(predictions, axis=0)
        StackingSubmission = pd.DataFrame({'predictions': predictions})
        StackingSubmission['Level'] = train_merge['Level']
        StackingSubmission.to_csv('Level.csv', sep=',', header=True, index=False, line_terminator="\n")

        #predictions[predictions > .5] = 1
        #predictions[predictions <= .5] = 0
        accuracy = sum(predictions == train_merge['Level']) / len(predictions)
        print("整体准确率为: ", accuracy)

        '''
        train_feature1 = train_merge[train_merge['Level']== 1]  # 全部的1训练集
        predictions1 = predictions[train_merge['Level']== 1]  # 截取browsed标签对应的位置的预测
        accuracy = sum(predictions1 == train_feature1['Level']) / len(predictions1)
        print('缺失1率：', 1 - accuracy)

        predictions2 = predictions[predictions == 1]  # 取出自己预测为1的坐标
        print('预测1的个数:' + str(len(predictions)))
        train_feature2 = train_merge[predictions == 1]  # 取出自己预测为1的位置 的 训练集的答案
        accuracy = sum(predictions2 == train_feature2['Level']) / len(predictions2)
        print('预测1准确率：', accuracy)
        end = time.time()
        print('total time is', end - start)
        '''
        print('===============================END=====================')
        exit()
    elif order == 'predict':
        list_two = ['train2','test2']
        for i in range(len(list_two)):
            # TODO 相似性
            cur = list_two[i]
            f1 = pd.read_csv('../2-feature/feature_collection_'+cur+'.csv', sep=',', low_memory=False)

            # TODO 读取特征
            type1_vecs = np.load('../2-feature/wv300_win100.'+cur+'_A_Title.npy')  # 10W行
            type2_vecs = np.load('../2-feature/wv300_win100.'+cur+'_R_Title.npy')
            type3_vecs = np.load('../2-feature/wv300_win100.'+cur+'_A_Content.npy')  # 10W行
            type4_vecs = np.load('../2-feature/wv300_win100.'+cur+'_R_Content.npy')

            # 变成
            feature = np.concatenate((type1_vecs, type2_vecs, type3_vecs, type4_vecs), axis=1)
            feature = reduce_pca(feature, 'c', 0.37)
            print('完成降维处理')
            train_feature = pd.DataFrame(feature)
            if i==0:
                feature_train = pd.concat([train_feature, f1], axis=1)
                print('训练数据')
                print(feature_train.info())
            else:
                feature_test = pd.concat([train_feature, f1], axis=1)
                print('测试数据')
                print(feature_test.info())

        #TODO 标签label
        train_merge = pd.read_csv('../1-prepare/train_merge.csv', sep=',', low_memory=False)
        label = train_merge['Level']
        label = label.astype(int)  # 将data变成Int的形式

        print('训练数据准备完成')

        # ---------------------------------
        print('预测中..')

        termob = classify.term()
        Level = termob.predict(feature_train, label,feature_test, 'gender')
        print('完成预测')

        test_merge = pd.read_csv('../1-prepare/test_merge.csv', sep=',', low_memory=False)
        StackingSubmission = pd.DataFrame({'Level': Level})
        StackingSubmission['L_ID'] = test_merge['L_ID']
        #StackingSubmission['jd_no'] = train_merge['jd_no']
        order =['L_ID','Level']
        StackingSubmission = StackingSubmission[order]
        StackingSubmission.to_csv('StackingSubmission.csv', sep=',', header=False, index=False, line_terminator="\n")
        # StackingSubmission = pd.DataFrame({'browsed': browsed, 'delivered':delivered,'satisfied':satisfied})
    elif order =='combine':
        #将type的向量和city edu向量结合起来
        merge = pd.read_csv('../1-prepare/train_merge_delivered.csv', sep=',', low_memory=False)

        # TODO 居住城市
        live_city = pd.get_dummies(merge['live_city_id'])
        # live_city = reduce_pca(live_city,'live_city')

        # TODO 教育程度:哑变量
        edu_cur = pd.get_dummies(merge['cur_degree_id'])
        # edu_cur = reduce_pca(edu_cur,'edu_cur')

        edu_min = pd.get_dummies(merge['min_edu_level'])
        # edu_min = reduce_pca(edu_min,'edu_min')

        f1 = pd.concat([edu_min, edu_cur, live_city], axis=1)
        f1 = np.array(f1)
        print('生成向量1')

        #TODO 读取type类型变量
        #type1_vecs = np.load('../2-find_feature/wv300_win100.train_delivered_type1.npy') #10W行
        type2_vecs = np.load('../2-find_feature/wv300_win100.train_delivered_type2.npy')

        #变成
        feature = np.concatenate((f1,type2_vecs),axis=1)
        feature = reduce_pca(feature, 'c', 0.9 )
        print('完成降维处理')

        #TODO 预测
        label_answer = pd.DataFrame(merge['delivered'])
        label = 'delivered'
        train_feature = pd.DataFrame(feature)
        # 进行三折运算
        kf = KFold(n_splits=3, shuffle=False, random_state=1)
        predictions = []
        for train, test in kf.split(train_feature):
            # The predictors we're using to train the algorithm.  Note how we only take then rows in the train folds.
            train_predictors = (train_feature.iloc[train, :])
            # The target we're using to train the algorithm.
            train_target = label_answer.iloc[train]
            test_predictions = classify.term().predict(train_predictors, train_target,
                                                       train_feature.iloc[test, :], 'gender')
            predictions.append(test_predictions)

        # 将结果写入csv
        predictions = np.concatenate(predictions, axis=0)
        StackingSubmission = pd.DataFrame({'predictions': predictions})
        #StackingSubmission[label] = label_answer[label]
        #StackingSubmission.to_csv(label + '.csv', sep=',', header=True, index=False, line_terminator="\n")

        predictions[predictions > .5] = 1
        predictions[predictions <= .5] = 0
        accuracy = sum(predictions == label_answer[label]) / len(predictions)
        print("整体准确率为: ", accuracy)

        label_answer1 = label_answer[label_answer[label] == 1]  # 全部的1训练集
        predictions1 = predictions[label_answer[label] == 1]  # 截取browsed标签对应的位置的预测
        accuracy = sum(predictions1 == label_answer1[label]) / len(predictions1)
        print('缺失1率：', 1 - accuracy)

        predictions2 = predictions[predictions == 1]  # 取出自己预测为1的坐标
        print('预测1的个数:' + str(len(predictions2)))
        label_answer2 = label_answer[predictions == 1]  # 取出自己预测为1的位置 的 训练集的答案
        accuracy = sum(predictions2 == label_answer2[label]) / len(predictions2)
        print('预测1准确率：', accuracy)

        end = time.time()
        print('total time is', end - start)

        print('===============================END=====================')
    end = time.time()
    print('total time is', end - start)
