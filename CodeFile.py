# -*- coding: utf-8 -*-
"""
@author: sujanbajracharya
"""

import pylab as pl
import numpy as np
import pandas as pd
import pandas_profiling as pp
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def getData(fileName, tabname):
    dct_all_sheet = pd.read_excel(fileName, sheet_name=None)
    dataframe = pd.DataFrame(dct_all_sheet[tabname])
    return dataframe


def profiledata(dataframe):
    profile = dataframe.profile_report()
    profile.to_file("DQAssessmentCustDemAdd.html")


def featureselection(dataframe):
    X = dataframe[[
                  'owns_car_encoded', 'job_industry_category_encoded','gender_encoded']]

    if 'Label' in dataframe.columns:
        Y = dataframe['Label']
    else:
        Y = []

    return X, Y

    # 'owns_car_encoded', 'job_industry_category_encoded' - .66
    # 'owns_car_encoded', 'job_industry_category_encoded','gender_encoded' -  0.7112
    # 'owns_car_encoded', 'job_industry_category_encoded','wealth_segment_encoded' - .63
    # 'owns_car_encoded', 'job_industry_category_encoded','wealth_segment_encoded','gender_encoded' - .6306
    # 'owns_car_encoded', 'wealth_segment_encoded','gender_encoded' - .64



def splitdata(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=0)
    return X_train, X_test, Y_train, Y_test


def trainlogisticmodel(X_train, Y_train):
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, Y_train)
    return logistic_regression


def getlogisticmodelprediction(logistic_regression, X_test):
    Y_pred = logistic_regression.predict(X_test)
    return Y_pred


def evaluatelogisticmodelprediction(Y_test, Y_pred):
    confusion_matrix = pd.crosstab(Y_test, Y_pred, rownames=[
                                   'Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    print('Accuracy: ', metrics.accuracy_score(Y_test, Y_pred))
    pl.show()


# Execution Steps:
traintestdata = getData(
    'KPMG_VI_New_raw_data_update_final.xlsx', 'TransCustDemAdd')
# profiledata(data)
X, Y = featureselection(traintestdata)
X_train, X_test, Y_train, Y_test = splitdata(X, Y)
logistic_regression = trainlogisticmodel(X_train, Y_train)
Y_pred = getlogisticmodelprediction(logistic_regression, X_test)
evaluatelogisticmodelprediction(Y_test, Y_pred)
newdata = getData('KPMG_VI_New_raw_data_update_final.xlsx', 'NewCustomerList')
X_new, Y_new = featureselection(newdata)
Y_pred_new = getlogisticmodelprediction(logistic_regression, X_new)
print(Y_pred_new.sum())
newdata['Recommendation'] =  Y_pred_new
newdata.to_csv('Recommendation.csv', index = True) 

