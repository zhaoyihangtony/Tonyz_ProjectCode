import pandas as pd 
import numpy as np 
#import scikits.statsmodels.api as sm


seasons = {'2012-13','2013-14','2014-15','2015-16','2016-17'}
#seasons = {'2016-17'}
#years = {'2013','2014','2015','2016','2017'}
foldername = 'predicte player/'
subfolder = 'perdef/'
#winratefolder = 'TrainData/'



for i in seasons:
    print(i)
    Teamdata = pd.read_csv(foldername+subfolder+i+'TeamTotalDEFPER.csv')
    winrate = pd.read_csv(foldername + i + 'TeamWinRate.csv')
    #print(winrate)
    df_winrate = pd.DataFrame(winrate)

    df = df_winrate[['TEAM','WIN%']]
    #print(df)

    df2 = pd.merge(Teamdata,df,on='TEAM')
    vdf = pd.DataFrame(df2)
    vdf.to_csv(foldername+subfolder+i+'TeamTotalPERDEF_Train.csv')
