import pandas as pd
import numpy as np


team=[] #list will hold teams as encountered in playersdef file. (length will be 30)
starterRPMTotal=[] #list will hold sum of rpm of startup players of teams (length will be 30) 
backupRPMTotal=[] #list will hold sum of rpm of backup players of teams (length will be 30) 
count=0
season='2017'

rpmfile='rpm'+season+'.csv' #file containing RPM data
playerfile='Playersdef'+Season+'.csv' #file contains players in a team data
rpmdf=pd.read_csv(rpmfile)
playersdf=pd.read_csv(playerfile)

def getRPM(target): # gets players rpm from rpmfile
    for index, row in rpmdf.iterrows():
        if (row['Players name']==target):
            rpm=row['Players RPM']
            return rpm
    return float(0)

for index,row in playersdf.iterrows():
    count+=1
    if (index%5==0): # every 5 rows are for a team, so it resets to 0 after counting 5.
        startupTotal=0
        backupTotal=0
    srpm=getRPM(row['startupPlayers']) #sends first row's starter player name to getRPM fuction to fetch RPM
    startupTotal+=srpm   # adds up start up players rpm
    brpm=getRPM(row['backupPlayers']) #sends first row's backup player name to getRPM fuction to fetch RPM
    backupTotal+=brpm # adds up back up players rpm
    
    if(count==5): #after 5 counts, startup total and backup total is added to two lists. this corresponds to a single team. 
        count=0
        team.append(row['Team'])
        starterRPMTotal.append(startupTotal)
        backupRPMTotal.append(backupTotal)



dic={'Team':team,'startupRPM':starterRPMTotal,'backupRPM':backupRPMTotal}
vdf=pd.DataFrame(dic)
filename='rpmtotalteamwise'+season+'.csv'
vdf.to_csv(filename, sep='\t')
