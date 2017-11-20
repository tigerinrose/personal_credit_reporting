#!usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from sklearn import metrics, cross_validation
import numpy as np
import pickle

# reload(sys) #
# import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('utf8')


# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


# def read_data(data_file):
#     import gzip
#     f = gzip.open(data_file, "rb")
#     print("====================")
#     print(f)
#     train, val, test = pickle.load(f, encoding='iso-8859-1')
#     print(train)
#     print(len(train))
#     # print(val)
#     # train, val, test = pickle.load(open(f, "rb"), encoding='iso-8859-1')
#     f.close()
#     train_x = train[0]
#     print(train_x)
#     print(train[1])
#     train_y = train[1]
#     test_x = test[0]
#     test_y = test[1]
#     return train_x, train_y, test_x, test_y

# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir


# x,y=getDataSet('iris.data.txt',4)
# tr1,tr2,ts1,ts2 = dataDiv(x,y)

def getDataSet(filename, numberOfFeature):  # 将数据集读入内存
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # get the number of lines in the file  file.readlines()是把文件的全部内容读到内存，并解析成一个list
    returnMat = zeros((numberOfLines, numberOfFeature-1))  # prepare matrix to return  3代表数据集中特征数目###
    classLabelVector = []  # prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # strip() 参数为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        #listFromLine = line.split(',')  # split 以什么为标准分割一次  分成数组中的每个元素
        listFromLine = line.split( )  # split 以什么为标准分割一次  分成数组中的每个元素
        returnMat[index, :] = listFromLine[0:numberOfFeature-1]
        # classLabelVector.append(int(listFromLine[-1]))   #append() 方法向列表的尾部添加一个新的元素
        # if listFromLine[-1] == 'Iris-setosa':
        #     classLabelVector.append(1)
        # elif listFromLine[-1] == 'Iris-versicolor':
        #     classLabelVector.append(2)
        # else:
        #     # elif listFromLine[-1] == 'Iris-virginica' :
        #     classLabelVector.append(3)
        if listFromLine[-1] == '1':
            classLabelVector.append(1)
        else:
            # elif listFromLine[-1] == '2' :
            classLabelVector.append(2)
        index += 1
    return returnMat, classLabelVector


def read_data(inMat, classVector):
# 方法一
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []
    # #读取前700条数据作为训练集，后300条数据作为测试集
    # for i in range(0,700)   :
    #     trainData.append(inMat[i][:])
    #     trainLabel.append(classVector[i])
    # for i in range(0,300)  :
    #     testData.append(inMat[700+i][:])
    #     testLabel.append(classVector[700+i])

    # #随机选取,训练集/测试集  800/200  包括重复值
    # for i in range(0, 800):
    #     j=random.randint(0, 1000 - 1)
    #     trainData.append(inMat[j][:])
    #     trainLabel.append(classVector[j])
    # for i in range(0, 200):
    #     j=random.randint(0, 1000 - 1)
    #     testData.append(inMat[j][:])
    #     testLabel.append(classVector[j])


    # # 随机选取,训练集/测试集  1000/500  包括重复值
    # for i in range(0, 1000):
    #     j = random.randint(0, 1000 - 1)
    #     trainData.append(inMat[j][:])
    #     trainLabel.append(classVector[j])
    # for i in range(0, 500):
    #     j = random.randint(0, 1000 - 1)
    #     testData.append(inMat[j][:])
    #     testLabel.append(classVector[j])


    # for i in range(0,160)   :
    #     trainData.append(inMat[i][:])
    #     trainLabel.append(classVector[i])
    # for i in range(0,160)    :
    #     trainData.append(inMat[200+i][:])
    #     trainLabel.append(classVector[200+i])
    # for i in range(0,160)    :
    #     trainData.append(inMat[400+i][:])
    #     trainLabel.append(classVector[400+i])
    # for i in range(0,160)    :
    #     trainData.append(inMat[600+i][:])
    #     trainLabel.append(classVector[600+i])
    # for i in range(0,160)    :
    #     trainData.append(inMat[800+i][:])
    #     trainLabel.append(classVector[800+i])
    #
    # for i in range(0,40)  :
    #     testData.append(inMat[160+i][:])
    #     testLabel.append(classVector[160+i])
    # for i in range(0,40)  :
    #     testData.append(inMat[360+i][:])
    #     testLabel.append(classVector[360+i])
    # for i in range(0,40)  :
    #     testData.append(inMat[560+i][:])
    #     testLabel.append(classVector[560+i])
    # for i in range(0,40)  :
    #     testData.append(inMat[760+i][:])
    #     testLabel.append(classVector[760+i])
    # for i in range(0,40)  :
    #     testData.append(inMat[960+i][:])
    #     testLabel.append(classVector[960+i])


    trainData, testData, trainLabel, testLabel =  cross_validation.train_test_split(inMat,classVector,test_size=0.3, random_state=0)

    return trainData, trainLabel, testData, testLabel

# x,y=getDataSet('german.data-numeric.txt',25)
# tr1,tr2,ts1,ts2 = read_data(x,y)
# print(tr1)
# print(tr2)
# print(ts1)
# print(ts2)


if __name__ == '__main__':
    data_file = "mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    # test_classifiers = ['LR', 'DT', 'GBDT']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }
    # classifiers = {'LR': logistic_regression_classifier,
    #                'DT': decision_tree_classifier,
    #                'GBDT': gradient_boosting_classifier
    #                }
    print('reading training and testing data...')
    #========================================================
    x, y = getDataSet('german.data-numeric.txt', 25)
    train_x, train_y, test_x, test_y = read_data(x,y)
    #========================================================
    # num_train, num_feat = train_x.shape
    # num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** Data Info *********************')
    print('#training data: %d, #testing_data: %d, dimension: %d' % (700, 300, 24))

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.2f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))  