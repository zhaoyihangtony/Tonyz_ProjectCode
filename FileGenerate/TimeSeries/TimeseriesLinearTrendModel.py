import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from statsmodels.tsa.stattools import adfuller

seasons = ['2011-12','2012-13','2013-14','2014-15','2015-16','2016-17']
for i in seasons:
    gamesfilename = 'nba_gamelog_ScoreBox'+ i + '.csv'
    gamesfile = pd.read_csv('FileGenerate/TimeSeries/' + gamesfilename)
    gamesfile.index = pd.PeriodIndex(gamesfile.Date,freq='D')
    #print(gamesfile.Date.min())
    gamesfile.Date = pd.to_datetime(gamesfile.Date)
    #print(gamesfile.dtypes)
    gamesfile['timeIndex'] = gamesfile.Date - gamesfile.Date.min()
    #print(gamesfile.head())
    #print(gamesfile.dtypes)
    gamesfile['timeIndex'] = gamesfile['timeIndex']/np.timedelta64(1,'D')
    gamesfile['timeIndex'] = gamesfile['timeIndex'].astype(int)
    trainfile = gamesfile.groupby('timeIndex')['Winning'].mean()
    trainfile = pd.DataFrame(trainfile)
    trainfile.columns = ['Winning']
    trainfile['timeIndex'] = trainfile.index
    
    print(trainfile)
    #trainfile.to_csv('FileGenerate/TimeSeries/traintest.csv')

    #plt.plot(trainfile.index,trainfile.Winning,color = 'red')
    #plt.show()

    model_linear = smf.ols('Winning ~ timeIndex', data = trainfile).fit()

    print(model_linear.summary())

    print(model_linear.params)
    model_linear_pred = model_linear.predict()
    print(model_linear_pred)
    trainfile.plot(kind= 'line',x='timeIndex',y='Winning')
    plt.plot(trainfile.timeIndex,model_linear_pred,'-')

    model_linear.resid.plot(kind = 'bar')
    plt.show()

    #print(gamesfile)
    #print(gamesfile.timeIndex.tail())
    
    

    #gamesfile.date = pd.DatetimeIndex(gamesfile.date)
    
    #print(gamesfile.index)
    '''
    x= gamesfile.index
    y= gamesfile.Winning
    plt.plot(x,y,'r')
    '''
    gamesfile.Winning.plot(color='red')
    #plt.plot('r')
    plt.xlabel('Date')
    plt.ylabel('Winning')
    plt.title('2011-12 result')


    #gamesfile.Winning.plot('rs')
    plt.show()