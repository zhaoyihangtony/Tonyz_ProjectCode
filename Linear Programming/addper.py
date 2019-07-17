import pandas as pd 
import numpy as np 

foldername = 'predicte player/perdef/'
seasons = {'2012-13','2013-14','2014-15','2015-16','2016-17'}

for i in seasons:
    perfile = pd.read_csv(foldername+'per'+i+'.csv')
    deffile = pd.read_csv(foldername+'Players_off_def_'+i+'.csv')
    #df = pd.DataFrame()
    #df2 = pd.DataFrame()
    PERstartup = []
    PERbackup = [] 
    for index,row in deffile.iterrows():
        
        df=perfile['Players PER'].loc[perfile['Players name'] == row['startupPlayers']]
        df2=perfile['Players PER'].loc[perfile['Players name'] == row['backupPlayers']]
        #print(np.sum(df))

        #startupper = df[['Players PER']]
        #backupper = df2[['Players PER']]

        #print(startupper['Players PER'])

        PERstartup.append(np.sum(df))
        PERbackup.append(np.sum(df2))
    deffile['PERstartup'] = PERstartup
    deffile['PERbackup'] = PERbackup

    deffile.to_csv(foldername+i+'perdefstats.csv')
    
    
