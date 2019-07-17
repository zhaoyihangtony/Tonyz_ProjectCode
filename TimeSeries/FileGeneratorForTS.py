from nba_py import game
from nba_py import Scoreboard
from nba_py import team
import pandas as pd

#seasons = ['2011-12','2012-13','2013-14','2014-15','2015-16','2016-17']
#month = [1,2,3,4,5,6,7,8,9,10,11,12]
seasons = ['2012-13','2013-14','2014-15','2015-16','2016-17']

for i in seasons:
    GameIDfile = pd.read_csv('FileGenerate/TimeSeries/' + 'nba_gamelog' + i +'.csv')
    Date = []
    HomeTeam = []
    VisitTeam = []
    HomeTeamPTS = []
    VisitTeamPTS = []
    Winning = []
    count =0
    for index, row in GameIDfile.iterrows():
        count+=1
        #print(row['Game_ID']
        print(count)
        gameid = str(row['Game_ID'])
        games = game.Boxscore('00'+gameid).team_stats()
        if(len(games)==2):
            Date.append(row['GAME_DATE'])
            HomeTeam.append(row['HomeTeam'])
            VisitTeam.append(row['VisitTeam'])
        
            if (row['HomeTeam']==games.TEAM_ABBREVIATION[0]):
                HomeTeamPTS.append(games.PTS[0])
                VisitTeamPTS.append(games.PTS[1])
                Winning.append(games.PTS[0]/games.PTS[1])
            else:
                HomeTeamPTS.append(games.PTS[1])
                VisitTeamPTS.append(games.PTS[0])
                Winning.append(games.PTS[1]/games.PTS[0])
    dic = {'Date': Date, 'HomeTeam': HomeTeam, 'VisitTeam': VisitTeam,
        'HomeTeamPTS': HomeTeamPTS, 'VisitTeamPTS': VisitTeamPTS,'Winning': Winning}
    vdf = pd.DataFrame(dic).set_index('Date')
    vdf.to_csv('FileGenerate/TimeSeries/' + 'nba_gamelog_ScoreBox' + i +'.csv')
    
            
            
        
'''
gameID = '21201213'
game = game.Boxscore('00'+gameID).team_stats()

print (len(game))  
'''    
