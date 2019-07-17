 import pandas as pd 
import numpy as np 

season = '2017'
filename = 'GamesPerDef' + season + '.csv'
file = pd.read_csv(filename)
streakcount = 10

teamname = []
for index, row in file.iterrows():
    if (row['HomeTeam'] not in teamname[:]):
        teamname.append(row['HomeTeam'])

for loop in range(3,streakcount+1):
    HomewinningStreakTeams = []
    VisitwinningStreakTeams = []
    winningStreakTeams = []
    countWinning = 0

    #countWinningArray = []
    #countWinningArrayTemple = []
    #countWinningArray.append('Winning Streak')

  # print(winningStreakTeams)
    for i in range(0,30):
        for index, row in file.iterrows():
 
            if (row['HomeTeam'] == teamname[i] or row['VisitTeam'] == teamname[i]):
                if(row['WoL'] == 'Win' and row['HomeTeam'] == teamname[i]):
                    if (countWinning >= loop):                        
                        winningStreakTeams.append(row)
                        #print(np.array(winningStreakTeams))
                    countWinning+= 1
                   # print(countWinning,count)
                    #print(np.array(winningStreakTeams))
                elif (row['WoL'] == 'Lose' and row['VisitTeam'] == teamname[i]):
                    countWinning+= 1
                    #print(countWinning,count)
                   # countWinningArrayTemple.append(countWinning)
                else:
                    countWinning = 0                                 
            else:
                pass

        if (countWinning <= loop):
            countWinning = 0

    vdf = pd.DataFrame(winningStreakTeams)
    name = str(loop) + 'HomeTeamwinningStreak.csv'
    vdf.to_csv(name)


for loop in range(3,streakcount+1):
    HomewinningStreakTeams = []
    VisitwinningStreakTeams = []
    winningStreakTeams = []
    countWinning = 0
    count = 0
    #countWinningArray = []
    #countWinningArrayTemple = []
    #countWinningArray.append('Winning Streak')

  # print(winningStreakTeams)
    for i in range(0,30):
        for index, row in file.iterrows():
#            count+= 1

            if (row['HomeTeam'] == teamname[i] or row['VisitTeam'] == teamname[i]):
                if(row['WoL'] == 'Win' and row['HomeTeam'] == teamname[i]):
                    countWinning+= 1
                    #print(countWinning,count)
                    #print(np.array(winningStreakTeams))
                elif (row['WoL'] == 'Lose' and row['VisitTeam'] == teamname[i]):
                    if (countWinning >= loop):                        
                        winningStreakTeams.append(row)
                    countWinning+= 1
                    #print(countWinning,count)
                    #print(np.array(winningStreakTeams))
                   # countWinningArrayTemple.append(countWinning)
                else:
                    countWinning = 0                                 
            else:
                pass

        if (countWinning <= loop):
            countWinning = 0

    vdf = pd.DataFrame(winningStreakTeams)
    name = str(loop) + 'VisitTeamwinningStreak.csv'
    vdf.to_csv(name)
