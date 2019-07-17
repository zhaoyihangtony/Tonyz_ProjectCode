import pandas as pd 
import numpy as np 


WoL = []#win or loose
homeTeam = []
homeTeamstartupPer = []
homeTeamstartupDef = []
homeTeambackupPer = []
homeTeambackupDef = []
visitTeam = []
visitTeamstartupPer = []
visitTeamstartupDef = []
visitTeambackupPer = []
visitTeambackupDef = []
season = '2013'

gamesfile = 'games' + season + '.csv'
game = pd.read_csv(gamesfile,sep='\t')
datafile = 'TotalPerDefofTeams' + season + '.csv'
data = pd.read_csv(datafile)

def getdata(teamname):
    for index,row in data.iterrows():
        if(teamname in 'NY Knicks'):
            teamname='new york'
        elif(teamname == 'SA'):
            teamname='san antonio'
        elif(teamname == 'OKC'):
            teamname='Oklahoma'
        elif(teamname == 'LAL'):
            teamname='los angeles'
        elif(teamname == 'BKN'):
            teamname='brooklyn'
        elif(teamname == 'SAC'):
            teamname='Sacramento'
        elif(teamname == 'LAC'):
            teamname='LA Clippers'
        elif(teamname == 'PHX'):
            teamname='Phoenix'
        elif(teamname == 'WSH'):
            teamname='Washington'
        elif(teamname == 'NO'):
            teamname='New Orleans'
        if ( teamname.lower() in row['Team'].lower()):
            return row['Team'],row['TotalbackupDef'],row['TotalbackupPer'],row['TotalstartupDef'],row['TotalstartupPer']
    return 0,0,0,0,0
            
          

for index,row in game.iterrows():
    (homeTeam_,homeTeambackupDef_,homeTeambackupPer_,homeTeamstartupDef_,homeTeamstartupPer_) = getdata(row['home_team'])
    (visitTeam_,visitTeambackupDef_,visitTeambackupPer_,visitTeamstartupDef_,visitTeamstartupPer_) = getdata(row['visit_team'])
    homeTeam.append(homeTeam_)
    homeTeambackupDef.append(homeTeambackupDef_)
    homeTeambackupPer.append(homeTeambackupPer_)
    homeTeamstartupDef.append(homeTeamstartupDef_)
    homeTeamstartupPer.append(homeTeamstartupPer_)
    visitTeam.append(visitTeam_)
    visitTeambackupDef.append(visitTeambackupDef_)
    visitTeambackupPer.append(visitTeambackupPer_)
    visitTeamstartupDef.append(visitTeamstartupDef_)
    visitTeamstartupPer.append(visitTeamstartupPer_)

    if (row['home_team_score']>row['visit_team_score']):
        win = 'Win'
        WoL.append(win)
    else:
        win = 'Lose'
        WoL.append(win)

dic = {'WoL':WoL,'HomeTeam':homeTeam,'HomeTeamBackupDef':homeTeambackupDef,'HomeTeambackupPer':homeTeambackupPer,'HomeTeamstartupDef':homeTeamstartupDef,'HomeTeamstartupPer':homeTeamstartupPer,'VisitTeam':visitTeam,'VisitBackupDef':visitTeambackupDef,'VisitTeambackupPer':visitTeambackupPer,'VisitTeamstartupDef':visitTeamstartupDef,'VisitTeamstartupPer':visitTeamstartupPer}
vdf = pd.DataFrame(dic)
filename = 'gamesPerDef' + season + '.csv'
vdf.to_csv(filename)





