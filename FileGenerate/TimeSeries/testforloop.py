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

    #print(gamesfile.WINrate.plot())
