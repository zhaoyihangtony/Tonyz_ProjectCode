import pandas as pd 
import numpy as np 


seasons = {'2012-13','2013-14','2014-15','2015-16','2016-17'}
#seasons = {'2016-17'}
#years = {'2013','2014','2015','2016','2017'}
foldername = 'predicte player/'


for i in seasons:
    print(i)
    playerStatsfile = pd.read_csv(foldername + i + 'PlayerStatsneeded.csv')
    pplosfile = pd.read_csv(foldername + 'Playersdef' + i + '.csv')
    df3 = pd.DataFrame()
    for index, row in pplosfile.iterrows():
        #if(row['Team']=='Oklahoma City Thunder'):
           # a=1
        #print(conut)
        #conut+=1
        SDEF = []
        BDEF = []
        
        try:
            #print(row['startupPlayers'])
            df = playerStatsfile.loc[playerStatsfile['PLAYER']==(row['startupPlayers'])]
            #print(df)

        except:
            print('cant find startupPlayers: '+ row['startupPlayers'])
        try:
            df2 = playerStatsfile.loc[playerStatsfile['PLAYER']==(row['backupPlayers'])]
            #print(df2)
        except:
            print('cant find backupPlayers: '+ row['backupPlayers'])
        
        SDEF.append(row['DEFstartup']) 
        BDEF.append(row['DEFbackup'])
        ##print(df2)
        df['DEF'] = SDEF
        df2['DEF'] = BDEF
        #print(df)
        #print(df2)
        df3=df3.append(df)
        df3=df3.append(df2)
        #print(df3)
    playerlist = df3.set_index('PLAYER')
    playerlist.to_csv(foldername+i+'PlayerlistRoster.csv')

