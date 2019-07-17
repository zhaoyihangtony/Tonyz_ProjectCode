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
    #teamstartup = {}
    #teamsbackup = {}
    teamname = []
    TOTALS_PTS = []
    TOTALS_AST = []
    TOTALS_STL = []
    TOTALS_BLK = []
    TOTALS_FGM = []
    TOTALS_FGA = []
    TOTALS_3PM = []
    TOTALS_3PA = []
    TOTALS_FTM = []
    TOTALS_FTA = []
    TOTALS_TOV = []
    TOTALS_PF = []
    TOTALS_OREB = []
    TOTALS_DREB = []
    TOTALS_REB = []
    TOTALS_DEF = []

    TOTALB_PTS = []
    TOTALB_AST = []
    TOTALB_STL = []
    TOTALB_BLK = []
    TOTALB_FGM = []
    TOTALB_FGA = []
    TOTALB_3PM = []
    TOTALB_3PA = []
    TOTALB_FTM = []
    TOTALB_FTA = []
    TOTALB_TOV = []
    TOTALB_PF = []
    TOTALB_OREB = []
    TOTALB_DREB = []
    TOTALB_REB = []
    TOTALB_DEF = []

    S_PTS = []
    S_AST = []
    S_STL = []
    S_BLK = []
    S_FGM = []
    S_FGA = []
    S_3PM = []
    S_3PA = []
    S_FTM = []
    S_FTA = []
    S_TOV = []
    S_PF = []
    S_OREB = []
    S_DREB = []
    S_REB = []
    S_DEF = []

    B_PTS = []
    B_AST = []
    B_STL = []
    B_BLK = []
    B_FGM = []
    B_FGA = []
    B_3PM = []
    B_3PA = []
    B_FTM = []
    B_FTA = []
    B_TOV = []
    B_PF = []
    B_OREB = []
    B_DREB = []
    B_REB = []
    B_DEF = []
    #conut = 0

    for index, row in pplosfile.iterrows():
        #if(row['Team']=='Oklahoma City Thunder'):
           # a=1
        #print(conut)
        #conut+=1
        
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
        
        S_PTS.append(np.array(df['PTS']))
        #print(S_PTS)
        S_AST.append(np.array(df['AST']))
        S_STL.append(np.array(df['STL']))
        S_BLK.append(np.array(df['BLK']))
        S_FGM.append(np.array(df['FGM']))
        S_FGA.append(np.array(df['FGA']))
        S_3PM.append(np.array(df['3PM']))
        S_3PA.append(np.array(df['3PA']))
        S_FTM.append(np.array(df['FTM']))
        S_FTA.append(np.array(df['FTA']))
        S_TOV.append(np.array(df['TOV']))
        S_PF.append(np.array(df['PF']))
        S_OREB.append(np.array(df['OREB']))
        S_DREB.append(np.array(df['DREB']))
        S_REB.append(np.array(df['REB']))
        S_DEF.append(np.array(row['DEFstartup']))

        
        S_PTS.append(np.array(df2['PTS']))
        #print(S_PTS)
        S_AST.append(np.array(df2['AST']))
        S_STL.append(np.array(df2['STL']))
        S_BLK.append(np.array(df2['BLK']))
        S_FGM.append(np.array(df2['FGM']))
        S_FGA.append(np.array(df2['FGA']))
        S_3PM.append(np.array(df2['3PM']))
        S_3PA.append(np.array(df2['3PA']))
        S_FTM.append(np.array(df2['FTM']))
        S_FTA.append(np.array(df2['FTA']))
        S_TOV.append(np.array(df2['TOV']))
        S_PF.append(np.array(df2['PF']))
        S_OREB.append(np.array(df2['OREB']))
        S_DREB.append(np.array(df2['DREB']))
        S_REB.append(np.array(df2['REB']))
        S_DEF.append(np.array(row['DEFbackup']))

        if ((index + 1) % 5 == 0):
            teamname.append(row['Team'])
            #print(row['Team'])
            #print(np.sum(S_PTS))
            TOTALS_PTS.append(np.sum(S_PTS))

            TOTALS_AST.append(np.sum(S_AST))
            TOTALS_STL.append(np.sum(S_STL))
            TOTALS_BLK.append(np.sum(S_BLK))
            TOTALS_FGM.append(np.sum(S_FGM))
            TOTALS_FGA.append(np.sum(S_FGA))
            TOTALS_3PM.append(np.sum(S_3PM))
            TOTALS_3PA.append(np.sum(S_3PA))
            TOTALS_FTM.append(np.sum(S_FTM))
            TOTALS_FTA.append(np.sum(S_FTA))
            TOTALS_TOV.append(np.sum(S_TOV))
            TOTALS_PF.append(np.sum(S_PF))
            TOTALS_OREB.append(np.sum(S_OREB))
            TOTALS_DREB.append(np.sum(S_DREB))
            TOTALS_REB.append(np.sum(S_REB))
            TOTALS_DEF.append(np.sum(S_DEF))

            '''
            TOTALB_PTS.append(np.sum(B_PTS))
            #print(np.sum(B_PTS))
            TOTALB_AST.append(np.sum(B_AST))
            TOTALB_STL.append(np.sum(B_STL))
            TOTALB_BLK.append(np.sum(B_BLK))
            TOTALB_FGM.append(np.sum(B_FGM))
            TOTALB_FGA.append(np.sum(B_FGA))
            TOTALB_3PM.append(np.sum(B_3PM))
            TOTALB_3PA.append(np.sum(B_3PA))
            TOTALB_FTM.append(np.sum(B_FTM))
            TOTALB_FTA.append(np.sum(B_FTA))
            TOTALB_TOV.append(np.sum(B_TOV))
            TOTALB_PF.append(np.sum(B_PF))
            TOTALB_OREB.append(np.sum(B_OREB))
            TOTALB_DREB.append(np.sum(B_DREB))
            TOTALB_REB.append(np.sum(B_REB))
            TOTALB_DEF.append(np.sum(B_DEF))
            '''

            del S_PTS[:]
            del S_AST[:]
            del S_STL[:]
            del S_BLK[:]
            del S_FGM[:]
            del S_FGA[:]
            del S_3PM[:]
            del S_3PA[:]
            del S_FTM[:]
            del S_FTA[:]
            del S_TOV[:]
            del S_PF[:]
            del S_OREB[:]
            del S_DREB[:]
            del S_REB[:]
            del S_DEF[:]

            del B_PTS[:]
            del B_AST[:]
            del B_STL[:]
            del B_BLK[:]
            del B_FGM[:]
            del B_FGA[:]
            del B_3PM[:]
            del B_3PA[:]
            del B_FTM[:]
            del B_FTA[:]
            del B_TOV[:]
            del B_PF[:]
            del B_OREB[:]
            del B_DREB[:]
            del B_REB[:]
            del B_DEF[:]

    dic = {'TEAM':teamname,
           'Total_PTS':TOTALS_PTS,
           'Total_AST':TOTALS_AST,
           'Total_STL':TOTALS_STL,
           'Total_BLK':TOTALS_BLK,
           'Total_FGM':TOTALS_FGM,
           'Total_FGA':TOTALS_FGA,
           'Total_3PM':TOTALS_3PM,
           'Total_3PA':TOTALS_3PA,
           'Total_FTM':TOTALS_FTM,
           'Total_FTA':TOTALS_FTA,
           'Total_TOV':TOTALS_TOV,
           'Total_PF':TOTALS_PF,
           'Total_OREB':TOTALS_OREB,
           'Total_DREB':TOTALS_DREB,
           'Total_DEF':TOTALS_DEF}
    vdf = pd.DataFrame(dic)
    print(vdf)
    vdf.to_csv(foldername+'Time/'+i+'TeamTotalstats(WITH_def).csv')
