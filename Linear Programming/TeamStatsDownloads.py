import pandas as pd 

seasons = {'2012-13','2013-14','2014-15','2015-16','2016-17'}
foldername = 'predicte player/'

for i in seasons:
    print(i)
    traditionalfilename = i+ 'playerTraditionalStats.csv'
    advancedfilename = i + 'playerAdvancedStats.csv'
    TraditionalFile = pd.read_csv(foldername + traditionalfilename)
    AdvancedFile = pd.read_csv(foldername + advancedfilename)
    df1 = TraditionalFile[['PLAYER','TEAM','GP','MIN','PTS','FGM','FGA','3PM','3PA','FTM','FTA','OREB','DREB','REB','AST','TOV','STL','BLK','PF']]
    df2 = AdvancedFile[['PLAYER','PACE']]
    #print(df2)
    df3 = pd.merge(df1,df2,on='PLAYER')
    #print(df3)
    file = pd.DataFrame(df3)
    file.to_csv(foldername + i +'PlayerStatsneeded.csv')
