# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:18:53 2021

@author: fbai_
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
import random as rn


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

from autogluon.tabular import TabularDataset, TabularPredictor


import tensorflow as tf
from keras import backend as K
import gc

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


import warnings 
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

os.environ["PYTHONWARNINGS"] = "ignore" 

np.random.seed(1234)
rn.seed(1234)
tf.random.set_seed(1234)

from seb_etfclsmc_level_pred import build_model, inference_model, AgRegression, LSTMRegression, create_dataset

# =================== constant ===================
file_path = './'
file_name = 'Vix9_level_1994_2021_X281.csv'
outfile_path = './results/'

# yearly test data [minus 1 from csv]
#test_year = 2010
#test_start = 4004
#test_stop = 4255

#test_year = 2011
#test_start = 4256
#test_stop = 4506
 
#test_year = 2012
#test_start = 4507
#test_stop = 4753
 
#test_year = 2013
#test_start = 4754
#test_stop = 5002
 
#test_year = 2014
#test_start = 5003
#test_stop = 5253

#test_year = 2015
#test_start = 5254
#test_stop = 5505

#test_year = 2016
#test_start = 5506
#test_stop = 5757

#test_year = 2017
#test_start = 5758
#test_stop = 6007

#test_year = 2018
#test_start = 6008
#test_stop = 6256

#test_year = 2019
#test_start = 6257
#test_stop = 6508

#test_year = 2020
#test_start = 6509
#test_stop = 6760

test_year = '2018-2020'
test_start = 6008
test_stop = 6760

feature_num = 278   
train_dataset = 4000 

algo = 'LSTM'

# ================== load data ================== 
test_dataset = test_stop-test_start+1

df = pd.DataFrame(pd.read_csv(file_path+file_name))

# training data
train_X = pd.DataFrame(df.iloc[(test_start-train_dataset-1):(test_start-1),0:(feature_num+1)]) 
train_Y = pd.DataFrame(df.iloc[(test_start-train_dataset-1):(test_start-1),(feature_num+1)]) 
train_Sig = pd.DataFrame(df.iloc[(test_start-train_dataset-1):(test_start-1),(feature_num+2)]) 

# this is time series, do not do train/valid split
#test_size = 0.1
#trn_X, val_X, trn_Y, val_Y = train_test_split(train_X, train_Y, test_size=test_size, random_state=1234)

# testing data
test_X = pd.DataFrame(df.iloc[(test_start-1):(test_stop),0:(feature_num+1)]) 
test_Y = pd.DataFrame(df.iloc[(test_start-1):(test_stop),(feature_num+1)]) 
test_Sig = pd.DataFrame(df.iloc[(test_start-1):(test_stop),(feature_num+2)]) 


# get date
train_date = train_X['X1']
test_date = test_X['X1'] 

# drop date from feature
train_X = train_X.drop(columns=['X1'])
test_X = test_X.drop(columns=['X1'])

# resert index 
data_train_X = (train_X.reset_index(drop=True))  
data_train_Y = (train_Y.reset_index(drop=True))  
data_train_date = (train_date.reset_index(drop=True)) 
data_train_Sig = (train_Sig.reset_index(drop=True)) 

data_test_X = (test_X.reset_index(drop=True)) 
data_test_Y = (test_Y.reset_index(drop=True)) 
data_test_date = (test_date.reset_index(drop=True)) 
data_test_Sig = (test_Sig.reset_index(drop=True)) 


# ================== Model training and prediction ==================
model = build_model(algo, data_train_X, data_train_Y)

data_pred_Y = inference_model(algo, model, data_test_X, data_test_Y)

# RMSE 
testScore = math.sqrt(mean_squared_error(data_test_Y, data_pred_Y))
print('algorithm = ', algo, ', Test Score: %.2f RMSE' % (testScore))


# ================== translate to signal prediction ================== 
data_pred_Sig = ['NONE'] * (test_dataset)   
data_test_Sig = (data_test_Sig['X281']).values.tolist()
testSignalPredError = [0] * (test_dataset)   

columns=['date','actual_return','pred_return','actual_sig','pred_sig']
test_results = pd.DataFrame(columns=columns)

for i in range(test_dataset):

    if i > 0:
        if (data_pred_Y[i] >= data_pred_Y[i-1])>0:
            data_pred_Sig[i] = "up"
        else:
            data_pred_Sig[i] = "down"
        
    else:
            data_pred_Sig[i] = "up"
    
    test_results.loc[i] = [data_test_date[i],data_test_Y['X280'][i],data_pred_Y[i],data_test_Sig[i],data_pred_Sig[i]] 

#accuracy = sum(1 for x,y in zip(data_pred_Sig,data_test_Sig) if x == y) / len(data_pred_Sig)
accuracy=accuracy_score(data_test_Sig, data_pred_Sig)



print('Predicted Signal accuracy: ', 100*accuracy,'%')

# save to csv
#print(test_results)
outfile_name = 'VIX level prediction by '+algo+' ('+str(test_year)+').csv'  
test_results.to_csv(outfile_path + outfile_name)
print(outfile_path + outfile_name," saved.")


# ================== plot ================== 
#plt.subplot(3, 1, 1)
#plt.plot(data_test_Y)
#plt.subplot(3, 1, 2)
#plt.plot(data_pred_Y)
#plt.subplot(3, 1, 3)
plt.title('VIX level prediction by ' + algo + ' (2018-2020)')
plt.xlabel("Data points")
plt.ylabel("VIX Level")
#plt.xticks(range(len(data_test_date)),data_test_date,size='small')
plt.plot(data_test_Y, label="label")
plt.plot(data_pred_Y, label="prediction")
plt.legend(loc="upper left")
plt.show()
