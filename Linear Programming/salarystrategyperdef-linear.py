from pulp import *
from functions import *

teamName=[]
playersName=[]
playersLineUpPosition=[]
playersPosition=[]
playersPER=[]
playersSalary=[]
playersDefenseRating=[]

salresult=open("salarystrategypernewdeflinear","w")

#defining constants

(w1,w2,w3,w4,intercept)=linearRegressionPerDef()
lineUpPositions=['S','B']
positionsCategory=['C','PF','PG','SF','SG']

(teamName,playersName,playersLineUpPosition,playersPosition,playersPER,playersSalary,playersCount,playersDefenseRating)=retrieveData(teamName,playersName,playersLineUpPosition,playersPosition,playersPER,playersSalary,playersDefenseRating)
perThreshold=perThresholdCalculation("perdef","linear","old")

prob = pulp.LpProblem("playersrecommendStrategy4lin", pulp.LpMinimize)

isIncluded=pulp.LpVariable.dicts("isIncluded",indexs=playersName,lowBound=0, upBound=1, cat='Integer')         

for lpos in lineUpPositions:
    for ppos in positionsCategory:
        (startIndex,endIndex)=returnIndex(ppos,lpos,playersLineUpPosition,playersPosition,playersCount)
        prob+=lpSum(isIncluded[playersName[player]] for player in range(startIndex,endIndex+1))==1


(starterSIndex,starterEIndex)=returnIndex(0,'S',playersLineUpPosition,playersPosition,playersCount)
(backupSIndex,backupEIndex)=returnIndex(0,'B',playersLineUpPosition,playersPosition,playersCount)

prob+=(w1*lpSum(isIncluded[playersName[player]]*playersPER[player] for player in range(starterSIndex,starterEIndex+1)))+(w2*lpSum(isIncluded[playersName[player]]*playersPER[player] for player in range(backupSIndex,backupEIndex+1)))+(w3*lpSum(isIncluded[playersName[player]]*playersDefenseRating[player] for player in range(starterSIndex,starterEIndex+1)))+(w4*lpSum(isIncluded[playersName[player]]*playersDefenseRating[player] for player in range(backupSIndex,backupEIndex+1)))+intercept >=perThreshold

prob+=lpSum(isIncluded[playersName[player]]*playersSalary[player] for player in range(playersCount))

status = prob.solve()

print "RECOMMENDED PLAYERS FOR MIN SALARY OBJECTIVE:"
salresult.write("RECOMMENDED PLAYERS FOR MIN SALARY OBJECTIVE:")
startupPER=0
backupPER=0
startupDEF=0
backupDEF=0
for i,player in enumerate(playersName):
    if(value(isIncluded[player])==1.0):
        if(playersLineUpPosition[i]=='S'):
            startupPER+=playersPER[i]
            startupDEF+=playersDefenseRating[i]
        else:
            backupPER+=playersPER[i]
            backupDEF+=playersDefenseRating[i]
        print "Name:",player,", LineupPosition:",playersLineUpPosition[i],", Position:",playersPosition[i],", PER:",playersPER[i],", Salary:",playersSalary[i],", Defense rating:",playersDefenseRating[i],", Current Team:",teamName[i]
        salresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))

print("\n")

percons=(w1*startupPER)+(w2*backupPER)+(w3*startupDEF)+(w4*backupDEF)+intercept
print "PER constraint:",percons
salresult.write("\n PER constraint: %f" % percons)
print "Minimum salary objective function value:",pulp.value(prob.objective)
salresult.write("\n Minimum salary objective function value: %f" % pulp.value(prob.objective))
salresult.close()
