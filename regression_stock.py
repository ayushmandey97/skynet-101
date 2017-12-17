'''

This algorithm is trained and tested on a google stock dataset from Quandl, so as to predict the change in adjusted close stock prices of the company's stock

'''

import numpy as np
import pandas as pd
import quandl, math, datetime
from time import time # for benchmarks

#for scaling features, to create train/test data, to compare benchmarks with svm, and regression
from sklearn import preprocessing , cross_validation, svm
from sklearn.linear_model import LinearRegression

#for visualisation
from matplotlib import pyplot as plt
from matplotlib import style

#for serializing python objects
import pickle

style.use('ggplot')

'''

Feature descriptions
------
High - low tells about volatility
open - starting price
close - ending price
open-close prices tells if price went up or down
vol - number of shares bought/sold
------

''' 

df = quandl.get('WIKI/GOOGL') #Using google stock dataset

#features
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']] #original features

#Introducing new features
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']) * 100
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

#Modified features for the dataframe
df = df[['Adj. Close',  'HL_PCT',  'PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
#df.fillna(-99999, inplace = False) #treat missing data as outlier

forecast_out = int(math.ceil(0.01*len(df))) #Forcasting 10% for number of data entries given

#Now label at every point is the forecast close price 10 (say days) into the future
df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:] #to predict, since we dont have y values for that
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2) # splitting 20% of total data into train and test sets

t = time()
#creating and fitting our regression model
clf = LinearRegression(n_jobs = -1) #to use most number of threads as possible
clf.fit(x_train, y_train)
print("Prediction time: ", round(time() - t,3), "s")

# Pickling for large datasets to avoid time-expensive re-training

# with open('linear_regression.pickle', 'wb') as f:
# 	pickle.dump(clf, f)
# pickle_in = open('linear_regression.pickle', 'rb')
# clf = pickle.load(pickle_in)

accuracy = round((clf.score(x_test, y_test)*100),2)
print()
print("Accuracy: " + str(accuracy) + "%")


#Predicting
forecast_set = clf.predict(x_lately)
print(" ------- Below is the forecase set --------")
print(forecast_set)


#Visualisation

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


