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
from pandas.tools.plotting import scatter_matrix


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
import matplotlib.dates as mdates
import seaborn as sns

# 
from logzero import logger
from logzero import logging


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# THIS FUNCTION CAN BE USED TO CREATE A TIME SERIES DATASET FROM ANY 1D ARRAY	


class StockNow:
    def __init__(self, ticker):
        self.ticker = ticker
    def get(self, item):
        today = datetime.now()
        panel_data = pd_data.get_data_yahoo(self.ticker, today, today, progress=False)
        stock_data = panel_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
        return stock_data[item]


# primary data anlysis

class StockDataGather:
    def __init__(self, ticker, start_date, end_date, dataDir="data"):
        self.ticker = ticker
        self.start_date = dtt.datetime.strptime(start_date, "%m-%d-%Y")
        self.end_date = dtt.datetime.strptime(end_date, "%m-%d-%Y")
        self.data_setup(dataDir)
        self.dataDir = dataDir + "/" + ticker
    def data_setup(self, dataDir):
        if not os.path.isdir(dataDir):
            os.mkdir(dataDir)
        if not os.path.isdir(dataDir + "/" + self.ticker):
            os.mkdir(dataDir + "/" + self.ticker)
        self.dataDir = dataDir + "/" + self.ticker
    def GetStockDataSheet(self, dataDir="data"):
        self.data_setup(dataDir)
        data_source = 'yahoo'
        fileName = "data/" + self.ticker + "/" + self.ticker + "-stock-data.csv"
        tkr = [ self.ticker ]
        if not os.path.exists(fileName):
            if (data_source == 'yahoo'):
                panel_data = pd_data.get_data_yahoo(tkr, self.start_date, \
                                                    self.end_date, progress=False)
                stock_data = panel_data.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']]
                stock_data.ffill()
                ma_day = [1,5,10]
                for ma in ma_day:
                    col_name = "MovAvg_" + str(ma)
                    stock_data[col_name] = panel_data.loc[:, ['Close']].rolling(window=ma, center=False).mean()
                stock_data.to_csv(fileName)
                logger.info("Wrote " + fileName)
            else:
                logger.error("Unsupported " + data_source)
        else:
            logger.info("Already exists " + fileName)



class StockDataRead:

    def __init__(self, ticker, dataDir="data/"):
        self.ticker = ticker
        self.fileName = dataDir + self.ticker + "/" + self.ticker + "-stock-data.csv"
        self.stock_data = pd.read_csv(self.fileName)

        # Minimum and maximum date in range
        self.min_date = dtt.datetime.strptime(min(self.stock_data['Date']), "%Y-%m-%d")
        self.max_date = dtt.datetime.strptime(max(self.stock_data['Date']), "%Y-%m-%d")
        self.stock_data['time_diff'] = self.stock_data['Date'].map(lambda x: (dtt.datetime.strptime(x, "%Y-%m-%d") - self.min_date)/timedelta(days=1))
        self.max_price = np.max(self.stock_data['Close'])
        self.max_price_date = self.stock_data[self.stock_data['Close'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]

        self.min_price = np.min(self.stock_data['Close'])
        self.min_price_date = self.stock_data[self.stock_data['Close'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]

        self.stock_data['min_diff'] = self.stock_data['Close'].map(lambda x: x - self.min_price)
        self.stock_data['max_diff'] = self.stock_data['Close'].map(lambda x: x - self.max_price)

    def StockFileName(self):
        return self.fileName

    def summary(self, predict_date):
        print("Data Collection ")
        print("\tStart date: " + str(self.min_date) )
        print("\tEnd date: " + str(self.max_date))

        date_format_f = '%Y-%m-%d'
        date_format_u = "%m-%d-%Y"
        p_day = datetime.strptime(predict_date, date_format_u)
        first_day = datetime.strptime(self.firstrow()['Date'], date_format_f)
        last_day = datetime.strptime(self.lastrow()['Date'], date_format_f)
        p_delta = p_day - last_day
        d_delta = first_day - last_day

        print("Data for " + str(self.rows()) + " weekdays " + str(d_delta.days - self.rows()) + " weekends")
        print("Data total for " + str(d_delta.days) + " days")
        print("First row Day " + str(self.firstrow()['Date']) + " closed at " + str(self.firstrow()['Close']) )
        print("Last row Day " + str(self.lastrow()['Date']) + " closed at " + str(self.lastrow()['Close']))
        print("Prediction day " + str(p_day))
        print("Last day to prediction day: " + str(p_delta.days) + " days")
        print("Max price: " + str(self.max_price) + "\t" + str(self.max_price_date))
        print("Min price: " + str(self.min_price) + "\t" + str(self.min_price_date))

    def rows(self):
        return self.stock_data.shape[0]

    def lastrow(self):
        return self.stock_data.iloc[-1]

    def firstrow(self):
        return self.stock_data.iloc[0]

    def predict(self, future_date):
        self.future_date = dtt.datetime.strptime(future_date, "%m-%d-%Y")
        td = (self.future_date - self.min_date)/timedelta(days=1)
        X_value = self.end_
        # 3 is where the time_diff is
        X_value[3] = td
        y_pred = self.algo_obj.predict([X_value])
        if (len(y_pred) > 0):
            return y_pred[0]
        else:
            return None

    def analyze(self, algo):
        if (algo == "linear"):
            self.linear()
        elif (algo == "svm"):
            self.svm()
        elif (algo == "randc"):
            self.rand_classifier()
        else:
            return None

    def scores(self, algo, y_test, y_pred):
        self.rmse = mean_squared_error(y_test, y_pred)
        self.r2_score = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')/100
        #print(algo + "\tMAE " + str(self.mae) + "\t:" + " mse: " + str(self.rmse) + " variance: %.2f" % self.r2_score)

    def linear(self):
        X, y = self.data_set_all()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred = regr.predict(X_test)
        self.scores("linear", y_test, y_pred)
        self.algo_obj = regr
        self.end_ = X_test[-1]


    def svm(self):
        X, y = self.data_set_all()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clf = svm.SVR(gamma="auto")
        parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)
        grid_obj = GridSearchCV(clf, param_grid=parameters, n_jobs=5, scoring=r2_scorer)
        grid_obj.fit(X_train, y_train)
        y_pred = grid_obj.predict(X_test)
        self.scores("svm", y_test, y_pred)
        self.algo_obj = grid_obj
        self.end_ = X_test[-1]


    def rand_classifier(self):
        X, y = self.data_set_all()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        y_train = np.array(y_train)
        y_train = y_train.astype(int)
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = np.array(clf.predict(X_test)).astype(float)
        self.scores("randc", y_test, y_pred)
        self.algo_obj = clf
        self.end_ = X_test[-1]

    def data_set_all(self):
        self.stock_data.dropna(axis=0, how='any', inplace=True)
        features = self.stock_data.columns[5:]
        X_all = self.stock_data[features].values
        y_all = list(self.stock_data['Close'])
        return X_all, y_all



class StockDataPlot:

    def __init__(self, FileName):

        self.data = pd.read_csv(FileName, \
                                usecols=['Date','Open', 'Close', 'Volume', 'MovAvg_10'],\
                                parse_dates=['Date'])
        daily_close = self.data[['Close']]
        daily_pct_change = daily_close.pct_change()
        self.data["daily"] = daily_pct_change
        min_periods = 10
        self.data["volatile"] = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)

        self.data.set_index('Date', inplace=True)

    def closing(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(11,7))
        top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
        bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
        top.plot(self.data.index, self.data['Close'], label="Close")
        top.plot(self.data.index, self.data['MovAvg_10'], label="M_avg_10")
        bottom.bar(self.data.index, self.data['Volume'])

        # set the labels
        top.axes.get_xaxis().set_visible(False)
        top.set_title('Closing')
        top.set_ylabel('Closing Price')
        bottom.set_ylabel('Volume')
        plt.show()


    def volume(self):
        fig, ax = plt.subplots(figsize=(15,7))
        self.data['Volume'].plot(grid=True, kind='barh', label="Volume", color='g', ax=ax)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.title("Volume")
        plt.show()

    def volatility(self, min_periods):
        fig, ax = plt.subplots(figsize=(15,7))
        vol = self.daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
        vol.plot(grid=True, kind='density', label="Volatility", ax=ax)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.legend(loc = 'upper right')
        plt.title("Volatility")
        plt.show()

    def scatter(self):
        #plt.figure
        fig, ax = plt.subplots(figsize=(11,5))
        with pd.plot_params.use('x_compat', True):
            self.data.daily.plot(color='r', label="Daily", ax=ax, linestyle='--', marker='o')
            self.data.volatile.plot(color='g', label="Volatile", ax=ax, linestyle='-', marker='o')
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.legend(loc = 'upper right')
        plt.show()

class wfc_dgather:

    def __init__(self, tkrs, begin_date, end_date):
        self.tkrs = tkrs
        self.begin = begin_date
        self.end = end_date

    def gather(self, location):
        logger.info("gather")
        for t in self.tkrs:
            sdg = StockDataGather(t, self.begin, self.end)
            sdg.GetStockDataSheet()

class wfc_dpredict:
    def __init__(self, cfg, tkr_name, future_date):
        self.cfg = cfg
        self.tkr_name = tkr_name
        self.future_date = future_date

    def predict(self):
        sdr = StockDataRead(self.tkr_name)
        sdr.analyze("linear")
        pval1 = sdr.predict(self.future_date)
        #print("Linear:\t" + str(self.future_date) + "\t" + str(pval1))
        sdr.analyze("svm")
        pval2 = sdr.predict(self.future_date)
        #print("svm:\t" + str(self.future_date) + "\t" + str(pval2))
        sdr.analyze("randc")
        pval3 = sdr.predict(self.future_date)
        #print("randc:\t" + str(self.future_date) + "\t" + str(pval3))
        return (pval1 + pval2 + pval3)/3


class wfc_dbenchmark:
    def __init__(self, cfg, pfname):
        self.cfg = cfg
        self.pfname = pfname

    def execute(self):
        pass

class wfc_dclean:
    def __init__(self, cfg):
        self.cfg = cfg

    def execute(self):
        loc = self.cfg.data_location()
        if os.path.isdir(loc):
        # TODO: use edgex_access to remove it
            shutil.rmtree(loc, ignore_errors=True)

class wfc_visual:
    def __init__(self, cfg, tkr_name):
        self.cfg = cfg
        self.tkr_name = tkr_name
    def now(self, item):
        sn = StockNow(self.tkr_name)
        return sn.get(item)

    def execute(self, cmd):
        sdr = StockDataRead(self.tkr_name)
        sdp = StockDataPlot(sdr.StockFileName())
        if (cmd == "closing"):
            sdp.closing()
        elif (cmd == "volume"):
            sdp.volume()
        elif (cmd == "daily"):
            sdp.scatter()
