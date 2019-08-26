# coding=utf-8
import multiprocessing
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
#from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold
#import xgboost as xgb
# from STFIWF import TfidfVectorizer #TODO??
from sklearn.feature_extraction.text import TfidfVectorizer #TODO??
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression,RidgeClassifier,PassiveAggressiveClassifier,Lasso,HuberRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,gradient_boosting
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler
class term(object):
    def __init__(self):
        random_rate = 8240

        clf1 = SGDClassifier(
            alpha=5e-05,
            average=False,
            class_weight='balanced',
            loss='log',
            n_iter=30,
            penalty='l2', n_jobs=-1, random_state=random_rate)
        clf2 = MultinomialNB(alpha=0.1)
        clf3 = LinearSVC(C=0.1, random_state=random_rate)
        clf4 = LogisticRegression(C=1.0,n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,solver='liblinear')
        clf5 = BernoulliNB(alpha=0.1)
        clf6 = VotingClassifier(estimators=[('sgd', clf1),
                                            ('mb', clf2),
                                            ('bb', clf3),
                                            ('lf', clf4),
                                            ('bnb', clf5)], voting='hard')
        clf7 = SGDClassifier(
            alpha=5e-05,
            average=False,
            class_weight='balanced',
            loss='log',
            n_iter=30,
            penalty='l1', n_jobs=-1, random_state=random_rate)
        clf8 = LinearSVC(C=0.9, random_state=random_rate)
        clf9 = LogisticRegression(C=0.5, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate)
        clf10 = MultinomialNB(alpha=0.9)
        clf11 = BernoulliNB(alpha=0.9)
        clf12 = LogisticRegression(C=0.2, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')
        clf13 = LogisticRegression(C=0.8, n_jobs=-1, max_iter=100, class_weight='balanced', random_state=random_rate,penalty='l1')
        clf14 = RidgeClassifier(alpha=8)
        clf15 = PassiveAggressiveClassifier(C=0.01, loss='squared_hinge', n_iter=20, n_jobs=-1)
        clf16 = RidgeClassifier(alpha=2)
        clf17 = PassiveAggressiveClassifier(C=0.5, loss='squared_hinge', n_iter=30, n_jobs=-1)
        clf18 = LinearSVC(C=0.5, random_state=random_rate)
        clf19 = MultinomialNB(alpha=0.5)
        clf20 = BernoulliNB(alpha=0.5)
        clf21 = Lasso(alpha=0.1, max_iter=20, random_state=random_rate)
        clf22 = Lasso(alpha=0.9, max_iter=30, random_state=random_rate)
        clf23 = PassiveAggressiveClassifier(C=0.1, loss='hinge', n_iter=30, n_jobs=-1, random_state=random_rate)
        clf24 = PassiveAggressiveClassifier(C=0.9, loss='hinge', n_iter=30, n_jobs=-1, random_state=random_rate)
        clf25 = HuberRegressor(max_iter=30)

        basemodel = [
            ['sgd', clf1],
            #['nb', clf2],
            ['lsvc1', clf3],
            ['LR1', clf4],
            ['bb',clf5],
            #['vote', clf6],
            ['sgdl1', clf7],
            ['lsvc2', clf8],
            ['LR2', clf9],
            #['nb2', clf10],
            ['bb2', clf11],
            ['LR3', clf12],
            ['LR4', clf13],
            ['rc1', clf14],
            ['pac1', clf15],
            ['rc2', clf16],
            ['pac2', clf17],
            ['lsvc3', clf18],
            #['nb3', clf19],
            ['bb3', clf20],
            ['lr5', clf21],
            ['lr6', clf22],
            ['rc3', clf23],
            ['pac3', clf24],
            ['hub', clf25],
        ]

        '''
        #random_state=0,n_estimators=300,min_samples_leaf=5,min_samples_split=10,max_depth=13
        clf1 =  RandomForestClassifier(random_state=0,n_estimators=300,min_samples_leaf=5,min_samples_split=10,max_depth=13) #SVM分类器
        clf2 = AdaBoostClassifier(n_estimators=500, learning_rate=0.01)
        clf3 = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
        clf4 = GradientBoostingClassifier(random_state=0,n_estimators=500, learning_rate=0.1, min_samples_split=3,
                                          min_samples_leaf=2, max_depth=5, verbose=0)
        clf5 = DecisionTreeClassifier(max_depth=8)
        clf6 = KNeighborsClassifier(n_neighbors=2)
        clf7 = SVC(kernel='linear', C=0.025)
        basemodel = [
                    ['rf',clf1],
                     #['ada', clf2],
                     #['et', clf3],
                     ['gb', clf4],
                     #['dt', clf5],
                     #['knn',clf6],
                     #['svm',clf7]
                     ]
        '''
        #####################################
        gbm = XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                            colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1)
        self.base_models = basemodel #基本模型
        self.LR=clf4 #逻辑回归
        #self.gbm = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=13,min_samples_leaf=2,min_samples_split=40) #SVM分类器
        self.gbm = GradientBoostingClassifier(random_state=0,n_estimators=500, learning_rate=0.1, min_samples_split=3,
                                          min_samples_leaf=2, max_depth=5, verbose=0)
    def stacking(self,X,Y,T,wv_X,wv_T,kind):
        """
        ensemble model:stacking

        """
        # TODO gender_traindatas, genderlabel, testdata, wv_gender_traindatas, w2vtest, 'gender'
        print ('fitting..')
        models = self.base_models
        folds = list(KFold(len(Y), n_folds=5, random_state=0))
        S_train = np.zeros((X.shape[0], len(models))) #有多少行 X.shape[0]
        S_test = np.zeros((T.shape[0], len(models)))

        for i, bm in enumerate(models):
            clf = bm[1] #模型是什么

            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = Y[train_idx]
                X_holdout = X[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:] #预测的结果
                S_train[test_idx, i] = y_pred #10W行的第i列
                S_test_i[:, j] = clf.predict(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        #print (S_train.shape,S_test.shape)

        S_train = np.concatenate((S_train,wv_X),axis=1)
        S_test = np.concatenate((S_test, wv_T), axis=1)

        print( S_train.shape,S_test.shape)

        print ('scalering..')
        min_max_scaler = StandardScaler()
        S_train = min_max_scaler.fit_transform(S_train)
        S_test = min_max_scaler.fit_transform(S_test)
        print( 'scalering over!')
        self.svc.fit(S_train, Y)
        yp= self.svc.predict(S_test)[:]
        return yp

    def validation(self, X, Y, wv_X, kind):
        """
        2-fold validation
        :param X: train text
        :param Y: train label
        :param wv_X: train wv_vec
        :param kind: age/gender/education
        :return: mean score of 2-fold validation
        """
        print( '向量化中...')
        X=np.array(X)
        fold_n=2
        folds = list(StratifiedKFold(Y, n_folds=fold_n, shuffle=False,random_state=0))
        score = np.zeros(fold_n)
        for j, (train_idx, test_idx) in enumerate(folds):
            print (j+1,'-fold')

            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_test = X[test_idx]
            y_test = Y[test_idx]

            wv_X_train =wv_X[train_idx]
            wv_X_test = wv_X[test_idx]

            vec = TfidfVectorizer(use_idf=True,sublinear_tf=False, max_features=50000, binary=True)
            vec.fit(X_train, y_train)
            X_train = vec.transform(X_train)
            X_test = vec.transform(X_test)

            print( 'shape',X_train.shape)

            ypre = self.stacking(X_train,y_train,X_test,wv_X_train,wv_X_test,kind)
            cur = sum(y_test == ypre) * 1.0 / len(ypre)
            score[j] = cur

        print( score)
        print (score.mean(),kind)
        return score.mean()

    def get_out_fold(self,clf, x_train, y_train, x_test,ntrain, ntest):
        # Some useful parameters which will come in handy later on
        SEED = 0  # for reproducibility
        NFOLDS = 3  # set folds for out-of-fold prediction
        kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            print(i)
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.fit(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    def predict(self,x_train,y_train,x_test,kind):
        """
        train and predict
        :param X: train text
        :param Y: train label
        :param T: test text
        :param wv_X: train wv
        :param wv_T: test wv
        :param kind: age/gender/education
        :return: array like ,predict of "kind"
        """
        #TODO gender_traindatas, genderlabel, testdata, wv_gender_traindatas, w2vtest, 'gender'
        print( 'predicting..向量化中...')
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        print(ntrain,ntest)

        x_train = x_train.values
        y_train = y_train.values
        x_test = x_test.values

        '''
        #对于机器学习模型开始遍历
        models = self.base_models
        list_train = []
        list_test = []
        for m in models:
            one_models = m[1]
            print('模型：'+m[0])
            oof_train, oof_test = self.get_out_fold(one_models,x_train, y_train, x_test, ntrain,ntest)
            list_train.append(oof_train)
            list_test.append(oof_test)

        #将list变成元组
        tup_train = tuple(list_train)
        #print(tup_train)
        tup_test = tuple(list_test)
        x_train = np.concatenate(tup_train,axis=1)
        x_test = np.concatenate(tup_test, axis=1)
        '''
        #print(x_train)
        #TODO 这里用 xgboost
        self.gbm.fit(x_train, y_train)
        predictions = self.gbm.predict(x_test)
        return predictions


