import pandas as pd 
import numpy as np 

team=[]
totalstartupPer = []
totalbackupPer = []
totalstartupDef = []
totalbackupDef = []
count=0
season='2013'

readfilename = 'PerDefa' + season + '.csv'
filename = pd.read_csv(readfilename)

for index,row in filename.iterrows():
    count+= 1
    if (index%5==0):
        startupPertotal = 0
        backupPertotal = 0
        startupDeftotal = 0
        backupDeftotal = 0
    startupPertotal+= row['startupPER']
    backupPertotal+= row['backupPER']
    startupDeftotal+= row['startupDEF']
    backupDeftotal+= row['backupDEF']

    if(count==5):
        count = 0
        team.append(row['Team'])
        totalstartupPer.append(startupPertotal)
        totalbackupPer.append(backupPertotal)
        totalstartupDef.append(startupDeftotal)
        totalbackupDef.append(backupDeftotal)

dic = {'Team':team,'TotalstartupPer':totalstartupPer,'TotalstartupDef':totalstartupDef,'TotalbackupPer':totalbackupPer,'TotalbackupDef':totalbackupDef}
vdf = pd.DataFrame(dic)
file1 = 'TotalPerDefofTeams'+ season +'.csv'
vdf.to_csv(file1)

    

 
