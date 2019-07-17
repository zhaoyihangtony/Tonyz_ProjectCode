from nba_py import team
import pandas as pd 
import numpy as np 

nbateamlog = pd.DataFrame()
#year = ('2016-17','2015-16')
year = ['2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17']
file = pd.read_csv('nbaTeamId.csv')
for i in year:
    for index,row in file.iterrows():

        nbateamlog = nbateamlog.append(team.TeamGameLogs(team_id=row['TEAM_ID'],season=i).info())
        HomeTeam = []
        VisitTeam = []
        VisitTeamID = []
        HomeTeamID = []
        GAME_DATE_fix = []
        WL = []

        for index,row in nbateamlog.iterrows():
    
            if (row['MATCHUP'].split(' vs. ') == [row['MATCHUP']]):
                HomeTeam.append(row['MATCHUP'].split(' @ ')[1])
                VisitTeam.append(row['MATCHUP'].split(' @ ')[0])
            else:
                HomeTeam.append(row['MATCHUP'].split(' vs. ')[0])
                VisitTeam.append(row['MATCHUP'].split(' vs. ')[1])



        nbateamlog.insert(4,"HomeTeam",HomeTeam)
        nbateamlog.insert(5,"VisitTeam",VisitTeam)
        '''
        for index,row in nbateamlog.iterrows():
            row['GAME_DATE'] = pd.to_datetime(row['GAME_DATE'])
            GAME_DATE_fix.append(pd.to_datetime(row['GAME_DATE']))
    
        for i,r in file.iterrows():
            if (row['HomeTeam']==r['ABBREVIATION']):
                HomeTeamID.append(r['TEAM_ID'])
            elif (row['VisitTeam']==r['ABBREVIATION']):
                VisitTeamID.append(r['TEAM_ID'])

        nbateamlog.insert(6,"HomeTeamID",HomeTeamID)
        nbateamlog.insert(7,"VisitTeamID",VisitTeamID)
        nbateamlog.insert(3,"GAME_DATE_FIX",GAME_DATE_fix)
        
        for index,row in nbateamlog.iterrows():
            if (row['Team_ID'] == row['HomeTeamID']):
                if(row['WL']=='L'):
                   WL.append('Lose')
                else:
                   WL.append('Win')
            else:
                if(row['WL']=='L'):
                   WL.append('Win')
                else:
                   WL.append('Lose')
    

        nbateamlog.insert(8,'HomeTeamWL',WL)
        '''
        #print(GAME_DATE_fix)
        #del nbateamlog['GAME_DATE']

        #print(nbateamlog) 
        nbateamlog.GAME_DATE = pd.to_datetime(nbateamlog.GAME_DATE)
        nbateamlog.index = pd.Index(nbateamlog.GAME_DATE)
        nbateamlog = nbateamlog.sort_values(by = 'GAME_DATE')

        nbateamlog = pd.DataFrame(nbateamlog).drop_duplicates(subset= 'Game_ID')  
        nbateamlog.to_csv('FileGenerate/TimeSeries/'+'nba_gamelog'+i+'.csv')
    


