import pandas as pd 
import numpy as np 

seasons = ['2011-12','2012-13','2013-14','2014-15','2015-16','2016-17']
for i in seasons:


    gamefile = pd.read_csv('FileGenerate/Winrate/nba_gamelog_withduplicate' + i + '.csv')
    Date = []
    GameID = []
    HomeTeam = []
    VisitTeam = []
    HomeTeamPTS = []
    VisitTeamPTS = []
    HomeTeamWinningrate = []
    VisitTeamWinningrate = []
    WL = []
    ratio = []

    for index,row in gamefile.iterrows():
        
        temp = gamefile.loc[gamefile['Game_ID']==row['Game_ID']]
        #print(temp.index[1])
        Date.append(temp.GAME_DATE[temp.index[0]])
        GameID.append(temp.Game_ID[temp.index[1]])
        HomeTeam.append(temp.HomeTeam[temp.index[0]])
        VisitTeam.append(temp.VisitTeam[temp.index[0]])
        HomeTeamW_PCT = 0
        VisitTeamW_PCT = 0
        if ('vs.' in temp.MATCHUP[temp.index[0]]):
            HomeTeamPTS.append(temp.PTS[temp.index[0]])
            VisitTeamPTS.append(temp.PTS[temp.index[1]])
            if (temp.WL[temp.index[0]]=='W'):
                HomeW = temp.W[temp.index[0]]-1
                HomeL = temp.L[temp.index[0]]
                HomeTotal = HomeW + HomeL
                if (HomeTotal == 0):
                    HomeTeamW_PCT=0
                else:
                    HomeTeamW_PCT = round(HomeW/HomeTotal,2)
                VisitW = temp.W[temp.index[1]]
                VisitL = temp.L[temp.index[1]]-1
                VisitTotal = VisitW + VisitL
                if(VisitTotal==0):
                    VisitTeamW_PCT=0
                else:
                    VisitTeamW_PCT = round(VisitW/VisitTotal,2)
                HomeTeamWinningrate.append(HomeTeamW_PCT)
                VisitTeamWinningrate.append(VisitTeamW_PCT)
            else:
                HomeW = temp.W[temp.index[0]]
                HomeL = temp.L[temp.index[0]]-1
                HomeTotal = HomeW + HomeL
                if (HomeTotal == 0):
                    HomeTeamW_PCT=0
                else:
                    HomeTeamW_PCT = round(HomeW/HomeTotal,2)
                VisitW = temp.W[temp.index[1]]-1
                VisitL = temp.L[temp.index[1]]
                VisitTotal = VisitW + VisitL
                if(VisitTotal==0):
                    VisitTeamW_PCT=0
                else:
                    VisitTeamW_PCT = round(VisitW/VisitTotal,2)
                HomeTeamWinningrate.append(HomeTeamW_PCT)
                VisitTeamWinningrate.append(VisitTeamW_PCT)
                #HomeTeamWinningrate.append(temp.W[temp.index[0]]/(temp.W[temp.index[0]]-1+temp.L[temp.index[0]]))
                #VisitTeamWinningrate.append((temp.W[temp.index[1]]-1)/(temp.W[temp.index[1]]+temp.L[temp.index[1]]-1))
                
            WL.append(round(temp.PTS[temp.index[0]]/temp.PTS[temp.index[1]],2))
            if(VisitTeamW_PCT==0):
                ratio.append(1)
            else:
                ratio.append(round(HomeTeamW_PCT/VisitTeamW_PCT,2))
        else:
            HomeTeamPTS.append(temp.PTS[temp.index[1]])
            VisitTeamPTS.append(temp.PTS[temp.index[0]])
            if (temp.WL[temp.index[0]]=='W'):
                HomeW = temp.W[temp.index[1]]
                HomeL = temp.L[temp.index[1]]-1
                HomeTotal = HomeW + HomeL
                if (HomeTotal == 0):
                    HomeTeamW_PCT=0
                else:
                    HomeTeamW_PCT = round(HomeW/HomeTotal,2)
                VisitW = temp.W[temp.index[0]]-1
                VisitL = temp.L[temp.index[0]]
                VisitTotal = VisitW + VisitL
                if(VisitTotal==0):
                    VisitTeamW_PCT=0
                else:
                    VisitTeamW_PCT = round(VisitW/VisitTotal,2)
                HomeTeamWinningrate.append(HomeTeamW_PCT)
                VisitTeamWinningrate.append(VisitTeamW_PCT)
            else:
                HomeW = temp.W[temp.index[1]]-1
                HomeL = temp.L[temp.index[1]]
                HomeTotal = HomeW + HomeL
                if (HomeTotal == 0):
                    HomeTeamW_PCT=0
                else:
                    HomeTeamW_PCT = round(HomeW/HomeTotal,2)
                VisitW = temp.W[temp.index[0]]
                VisitL = temp.L[temp.index[0]]-1
                VisitTotal = VisitW + VisitL
                if(VisitTotal==0):
                    VisitTeamW_PCT=0
                else:
                    VisitTeamW_PCT = round(VisitW/VisitTotal,2)
                HomeTeamWinningrate.append(HomeTeamW_PCT)
                VisitTeamWinningrate.append(VisitTeamW_PCT)
            WL.append(round(temp.PTS[temp.index[1]]/temp.PTS[temp.index[0]],2))
            if(VisitTeamW_PCT==0):
                ratio.append(1)
            else:
                ratio.append(round(HomeTeamW_PCT/VisitTeamW_PCT,2))
    print(len(Date),len(GameID),len(HomeTeam),len(VisitTeam),len(HomeTeamPTS))
        #print(WL,ratio)
    dic = {'Date': Date,'GameID': GameID, 'HomeTeam': HomeTeam, 'VisitTeam': VisitTeam,
        'HomeTeamPTS': HomeTeamPTS, 'VisitTeamPTS': VisitTeamPTS,'WL': WL,'HomeTeamW_PCT':HomeTeamWinningrate,'VisitTeamW_PCT':VisitTeamWinningrate,'Ratio':ratio}
    vdf = pd.DataFrame(dic).set_index('Date').drop_duplicates(subset= 'GameID')
    vdf.to_csv('FileGenerate/Winrate/' + 'nba_gamelog_ratio_fixedrate' + i +'.csv')