# TODO: a very simple general LSTM pulled from the internet. Needs to be re-written 
#
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import sys
import os
import shutil
import datetime as dtt
from datetime import timedelta
from datetime import datetime

from string import Template


import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pd_data
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv


import fix_yahoo_finance as yf
yf.pdr_override()

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.model_selection import GridSearchCV
from sklearn import svm, metrics, preprocessing
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation

import matplotlib.pyplot as plt
import visualize

# 
from logzero import logger
from logzero import logging

import math

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
    data_X, data_Y = [], []
    for i in range(len(dataset)-step_size-1):
        a = dataset[i:(i+step_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY

class StockLSTM():

    def __init__(self, fileName):
        dataset = pd.read_csv(fileName, usecols=[1,2,3,4])
        self.obs = np.arange(1, len(dataset) + 1, 1)
        self.OHLC_avg = dataset.mean(axis = 1)
        self.HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
        self.close_val = dataset[['Close']]

    def main(self):
        self.OHLC_avg = np.reshape(self.OHLC_avg.values, (len(self.OHLC_avg),1)) # 1664
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.OHLC_avg = scaler.fit_transform(self.OHLC_avg)

        # TRAIN-TEST SPLIT
        train_OHLC = int(len(self.OHLC_avg) * 0.75)
        test_OHLC = len(self.OHLC_avg) - train_OHLC
        train_OHLC, test_OHLC = self.OHLC_avg[0:train_OHLC,:], self.OHLC_avg[train_OHLC:len(self.OHLC_avg),:]

        # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
        trainX, trainY = new_dataset(train_OHLC, 1)
        testX, testY = new_dataset(test_OHLC, 1)

        # RESHAPING TRAIN AND TEST DATA
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        step_size = 1

        # LSTM MODEL
        model = Sequential()
        model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
        model.add(LSTM(16))
        model.add(Dense(1))
        model.add(Activation('linear'))

        # MODEL COMPILING AND TRAINING
        #model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad 
        model.compile(loss='mean_squared_error', optimizer='adam')
        #model.compile(loss='mean_squared_error', optimizer='SGD')
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)

        # PREDICTION
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # DE-NORMALIZING FOR PLOTTING
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        # TRAINING RMSE
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train RMSE: %.2f' % (trainScore))

        # TEST RMSE
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test RMSE: %.2f' % (testScore))


        # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
        self.trainPredictPlot = np.empty_like(self.OHLC_avg)
        self.trainPredictPlot[:, :] = np.nan
        self.trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

        # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
        self.testPredictPlot = np.empty_like(self.OHLC_avg)
        self.testPredictPlot[:, :] = np.nan
        self.testPredictPlot[len(trainPredict)+(step_size*2)+1:len(self.OHLC_avg)-1, :] = testPredict

        # DE-NORMALIZING MAIN DATASET
        self.OHLC_avg = scaler.inverse_transform(self.OHLC_avg)

