import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA #导入PCA算法库
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

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


'''

from sklearn.feature_extraction.text import TfidfVectorizer #TODO??
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression,RidgeClassifier,PassiveAggressiveClassifier,Lasso,HuberRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,gradient_boosting
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
'''
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 500)


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    print("调用特征提取函数")
    '''
    print("KNN")
    ada_est = KNeighborsClassifier()
    #{'leaf_size': 10, 'n_neighbors': 2, 'weights': 'uniform'}
    ada_param_grid = {'n_neighbors': [2], 'leaf_size': [2,10],'weights':['uniform','distance']}
    ada_grid = GridSearchCV(ada_est, ada_param_grid, n_jobs=-1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    exit()
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))
    '''

    # randomforest
    print('================randomforest==================')
    rf_est = RandomForestClassifier(random_state=0,n_estimators=50,max_depth=13,min_samples_leaf=2,min_samples_split=40)
    rf_param_grid = {'max_depth':range(2,14,1)}
    rf_grid = GridSearchCV(rf_est, rf_param_grid, n_jobs=1, cv=3, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)

    # 计算
    #y_predprob = rf_grid.predict_proba(titanic_train_data_X)[:, 1]
    #print("AUC Score (Train): %f" % metrics.roc_auc_score(titanic_train_data_Y, y_predprob))
    means = rf_grid.cv_results_['mean_test_score']
    params = rf_grid.cv_results_['params']
    for i, j in zip(means, params):
        print(i, j)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))

    '''
    print("====================GradientBoosting=========================")
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators':[60]}
    gb_grid = GridSearchCV(gb_est, gb_param_grid, n_jobs=1 , cv=3, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    y_predprob = gb_grid.predict_proba(titanic_train_data_X)[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(titanic_train_data_Y, y_predprob))
    means = gb_grid.cv_results_['mean_test_score']
    params = gb_grid.cv_results_['params']
    for i, j in zip(means, params):
        print(i, j)

    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))
    '''

    '''
    # AdaBoost
    print("===================AdaBoost=========================")
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [300], 'learning_rate': [0.1]}
    ada_grid = GridSearchCV(ada_est, ada_param_grid, n_jobs=1, cv=3, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Features from Ada Classifier:')
    print(str(features_top_n_ada[:10]))


    # ExtraTree
    print("======================ExtraTree=======================")
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [300], 'min_samples_split': [4], 'max_depth': [20]}
    et_grid = GridSearchCV(et_est, et_param_grid, n_jobs=1, cv=3, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best DT Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))


    # DecisionTree
    print("=========================DecisionTree=============================")
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [4], 'max_depth': [20]}
    dt_grid = GridSearchCV(dt_est, dt_param_grid, n_jobs=-1, cv=3, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Bset DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))
    '''

    # merge the three models
    # pd.concat，会按照列进行排序，drop_duplicates取出列的重复项
    # merge the three models
    # features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt
    features_top_n = pd.concat(
        [features_top_n_rf],
        ignore_index=True).drop_duplicates()

    # feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,feature_imp_sorted_gb, feature_imp_sorted_dt
    features_importance = pd.concat([feature_imp_sorted_rf,
                                     ], ignore_index=True)

    return features_top_n, features_importance


# 读取数据
train_merge = pd.read_csv('../1-prepare/train_merge.csv', sep=',', low_memory=False)
# train_feature = train_feature.iloc[0:10000, :]
# print(train_feature.columns)
# print('训练数据读取完毕')

# TODO 相似性
f1 = pd.read_csv('feature_collection_train2.csv', sep=',', low_memory=False)

# TODO 读取特征
type1_vecs = np.load('wv300_win100.train_A_Title.npy')  # 10W行
type2_vecs = np.load('wv300_win100.train_R_Title.npy')
type3_vecs = np.load('wv300_win100.train_A_Content.npy')  # 10W行
type4_vecs = np.load('wv300_win100.train_R_Content.npy')

# 变成
feature = np.concatenate((type1_vecs, type2_vecs, type3_vecs, type4_vecs), axis=1)
feature = reduce_pca(feature, 'c', 0.38)
print('完成降维处理')
train_feature = pd.DataFrame(feature)
train_data_X = pd.concat([train_feature, f1], axis=1)
print(train_data_X.info())

train_data_Y = train_merge['Level']  #
feature_to_pick = 20  # 挑选出几个特征
# TODO 查看数据是否正确
# print(train_data_X.head())
# exit()
features_top_n, features_importance = get_top_n_features(train_data_X, train_data_Y, feature_to_pick)
print('=======结束训练======')
print(features_top_n)
print('*' * 40)
print(features_importance.head(100))
