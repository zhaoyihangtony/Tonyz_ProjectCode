from pulp import *
from functions import *

teamName=[]
playersName=[]
playersLineUpPosition=[]
playersPosition=[]
playersPER=[]
playersSalary=[]
playersDefenseRating=[]

perresult=open("pernewdefstrategylogistic","w")
#defining constants

(w1,w2,w3,w4,intercept)=logisticRegressionPerDef()
print "printing coefficients"
print "\n w1=",w1," w2=",w2," w3=",w3," w4=",w4," intercept=",intercept
perresult.write("Model parameters are:") 
perresult.write("\n w1: %f, w2: %f, w3: %f, w4: %f, intercept: %f" %(w1,w2,w3,w4,intercept))

SalaryLimit=94140000
lineUpPositions=['S','B']
positionsCategory=['C','PF','PG','SF','SG']

(teamName,playersName,playersLineUpPosition,playersPosition,playersPER,playersSalary,playersCount,playersDefenseRating)=retrieveData(teamName,playersName,playersLineUpPosition,playersPosition,playersPER,playersSalary,playersDefenseRating)


#defining lp problem
prob_max = pulp.LpProblem("playersrecommendStrategy3logmax", pulp.LpMaximize)

#defining binary variable
isIncluded_max=pulp.LpVariable.dicts("isIncluded_max",indexs=playersName,lowBound=0, upBound=1, cat='Integer')         

# assigning constraints
for lpos in lineUpPositions:
    for ppos in positionsCategory:
        (startIndex,endIndex)=returnIndex(ppos,lpos,playersLineUpPosition,playersPosition,playersCount)
        prob_max+=lpSum(isIncluded_max[playersName[player]] for player in range(startIndex,endIndex+1))==1

prob_max+=lpSum(isIncluded_max[playersName[player]]*playersSalary[player] for player in range(playersCount))<=SalaryLimit

#index of starters and backup players
(starterSIndex,starterEIndex)=returnIndex(0,'S',playersLineUpPosition,playersPosition,playersCount)
(backupSIndex,backupEIndex)=returnIndex(0,'B',playersLineUpPosition,playersPosition,playersCount)

#assigning objective
prob_max+=(w1*lpSum(isIncluded_max[playersName[player]]*playersPER[player] for player in range(starterSIndex,starterEIndex+1)))+(w2*lpSum(isIncluded_max[playersName[player]]*playersPER[player] for player in range(backupSIndex,backupEIndex+1)))+(w3*lpSum(isIncluded_max[playersName[player]]*playersDefenseRating[player] for player in range(starterSIndex,starterEIndex+1)))+(w4*lpSum(isIncluded_max[playersName[player]]*playersDefenseRating[player] for player in range(backupSIndex,backupEIndex+1)))+intercept

# solve the problem
status = prob_max.solve()

print "RECOMMENDED PLAYERS FOR MAX OBJECTIVE:"
perresult.write("RECOMMENDED PLAYERS FOR MAX OBJECTIVE:")
totalSalaryCost_max=0
for i,player in enumerate(playersName):
    if(value(isIncluded_max[player])==1.0):
        totalSalaryCost_max+=playersSalary[i]
        print "Name:",player,", LineupPosition:",playersLineUpPosition[i],", Position:",playersPosition[i],", PER:",playersPER[i],", Salary:",playersSalary[i],", Defense rating:",playersDefenseRating[i],", Current Team:",teamName[i]
        perresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))
        

print "\nMaximum Objective function value:",sigmoid(pulp.value(prob_max.objective))
perresult.write("\nMaximum Objective function value: %f" % sigmoid(pulp.value(prob_max.objective)))
print "\nTotal Salary cost:",totalSalaryCost_max
perresult.write("\nTotal Salary cost: %f" % totalSalaryCost_max)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#minimize objective

#defining lp problem
prob_min = pulp.LpProblem("playersrecommendStrategy3logmin", pulp.LpMinimize)

#defining binary variable
isIncluded_min=pulp.LpVariable.dicts("isIncluded_min",indexs=playersName,lowBound=0, upBound=1, cat='Integer')         

# assigning constraints
for lpos in lineUpPositions:
    for ppos in positionsCategory:
        (startIndex,endIndex)=returnIndex(ppos,lpos,playersLineUpPosition,playersPosition,playersCount)
        prob_min+=lpSum(isIncluded_min[playersName[player]] for player in range(startIndex,endIndex+1))==1

prob_min+=lpSum(isIncluded_min[playersName[player]]*playersSalary[player] for player in range(playersCount))<=SalaryLimit

#index of starters and backup players
(starterSIndex,starterEIndex)=returnIndex(0,'S',playersLineUpPosition,playersPosition,playersCount)
(backupSIndex,backupEIndex)=returnIndex(0,'B',playersLineUpPosition,playersPosition,playersCount)

#assigning objective
prob_min+=(w1*lpSum(isIncluded_min[playersName[player]]*playersPER[player] for player in range(starterSIndex,starterEIndex+1)))+(w2*lpSum(isIncluded_min[playersName[player]]*playersPER[player] for player in range(backupSIndex,backupEIndex+1)))+(w3*lpSum(isIncluded_min[playersName[player]]*playersDefenseRating[player] for player in range(starterSIndex,starterEIndex+1)))+(w4*lpSum(isIncluded_min[playersName[player]]*playersDefenseRating[player] for player in range(backupSIndex,backupEIndex+1)))+intercept

# solve the problem
status = prob_min.solve()


print "RECOMMENDED PLAYERS FOR MIN OBJECTIVE:"
perresult.write("\n\nRECOMMENDED PLAYERS FOR MIN OBJECTIVE:")
totalSalaryCost_min=0
for i,player in enumerate(playersName):
    if(value(isIncluded_min[player])==1.0):
        totalSalaryCost_min+=playersSalary[i]
        print "Name:",player,", LineupPosition:",playersLineUpPosition[i],", Position:",playersPosition[i],", PER:",playersPER[i],", Salary:",playersSalary[i],", Defense rating:",playersDefenseRating[i],", Current Team:",teamName[i]
        perresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))

print "\nMinimum Objective function value:",sigmoid(pulp.value(prob_min.objective))
perresult.write("\nMinimum Objective function value: %f" % sigmoid(pulp.value(prob_min.objective)))
print "\nTotal Salary cost:",totalSalaryCost_min
perresult.write("\nTotal Salary cost: %f" % totalSalaryCost_min)

perresult.close()