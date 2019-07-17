import pandas as pd 

teams=[]
startupPlayer = []
backupPlayer = []
startupDef = []
backupDef = []
startupPer = []
backupPer = []

season='2013'

perdfile = 'per' + season + '.csv'
deffile = 'Playersdef' + season + '.csv'
playersdef = pd.read_csv(deffile)
playersper = pd.read_csv(perdfile)

def getPER(Playersname) :
    for index,row in playersper.iterrows():
        if (row['Players name'] == Playersname):
            Per = row['Players PER']
            return Per
    return float(0)

for index,row in playersdef.iterrows():
    teams.append(row['Team'])
    startupPlayer.append(row['startupPlayers'])
    backupPlayer.append(row['backupPlayers'])
    startupDef.append(row['DEFstartup'])
    backupDef.append(row['DEFbackup'])
    sPer = getPER(row['startupPlayers'])
    bPer = getPER(row['backupPlayers'])
    startupPer.append(sPer)
    backupPer.append(bPer)

dic={'Team':teams,'backupPER':backupPer,'backupDEF':backupDef,'backuplayers':backupPlayer,'startupPER':startupPer,'startupDEF':startupDef,'startuplayers':startupPlayer}
vdf = pd.DataFrame(dic)
filename = 'PerDefa'+ season +'.csv'
vdf.to_csv(filename)

