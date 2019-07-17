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
        if ('vs.' in temp.MATCHUP[temp.index[0]]):
            HomeTeamPTS.append(temp.PTS[temp.index[0]])
            VisitTeamPTS.append(temp.PTS[temp.index[1]])
            HomeTeamWinningrate.append(temp.W_PCT[temp.index[0]])
            VisitTeamWinningrate.append(temp.W_PCT[temp.index[1]])
            WL.append(round(temp.PTS[temp.index[0]]/temp.PTS[temp.index[1]],2))
            #if()
            ratio.append(round(temp.W_PCT[temp.index[0]]/temp.W_PCT[temp.index[1]],2))
        else:
            HomeTeamPTS.append(temp.PTS[temp.index[1]])
            VisitTeamPTS.append(temp.PTS[temp.index[0]])
            HomeTeamWinningrate.append(temp.W_PCT[temp.index[1]])
            VisitTeamWinningrate.append(temp.W_PCT[temp.index[0]])
            WL.append(round(temp.PTS[temp.index[1]]/temp.PTS[temp.index[0]],2))
            ratio.append(round(temp.W_PCT[temp.index[1]]/temp.W_PCT[temp.index[0]],2))
    print(len(Date),len(GameID),len(HomeTeam),len(VisitTeam),len(HomeTeamPTS))
        #print(WL,ratio)
    dic = {'Date': Date,'GameID': GameID, 'HomeTeam': HomeTeam, 'VisitTeam': VisitTeam,
        'HomeTeamPTS': HomeTeamPTS, 'VisitTeamPTS': VisitTeamPTS,'WL': WL,'HomeTeamW_PCT':HomeTeamWinningrate,'VisitTeamW_PCT':VisitTeamWinningrate,'Ratio':ratio}
    vdf = pd.DataFrame(dic).set_index('Date').drop_duplicates(subset= 'GameID')
    vdf.to_csv('FileGenerate/Winrate/' + 'nba_gamelog_ratio' + i +'.csv')
    
       
        
             


