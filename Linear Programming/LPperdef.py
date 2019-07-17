from sklearn import linear_model
import math
from pulp import *
import pandas as pd
import numpy as np  
import statsmodels.api as sm

def linearRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['TotalB_DEF',
          'TotalB_PER',
          'TotalS_DEF',
          'TotalS_PER']]
    testX=testfile[['TotalB_DEF',
          'TotalB_PER',
          'TotalS_DEF',
          'TotalS_PER']]
    Y=filename['WIN%']
    testY=testfile['WIN%']
    lm=linear_model.LinearRegression()
    model = lm.fit(X,Y)
    linearmodelpredict = lm.predict(testX)
    w1 = lm.coef_[0]
    w2 = lm.coef_[1]
    w3 = lm.coef_[2]
    w4 = lm.coef_[3]
    return w1,w2,w3,w4,lm.intercept_,linearmodelpredict

def LogisticRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['TotalB_DEF',
          'TotalB_PER',
          'TotalS_DEF',
          'TotalS_PER']]
    testX=testfile[['TotalB_DEF',
          'TotalB_PER',
          'TotalS_DEF',
          'TotalS_PER']]
    Y=filename['WIN%']
    testY=testfile['WIN%']
    X['one']= 1
    testX['one']=1
    mod = sm.Logit(Y,X)
    res = mod.fit()
    w1 = res.params[0]
    w2 = res.params[1]
    w3 = res.params[2]
    w4 = res.params[3]
    w0 = res.params[4]
    #print(res.params)
    logisticmodelpredict = res.predict(testX)
    
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w0,logisticmodelpredict
def returnindex (lpos,ppos,filename):
    Playerindex = []
    for index,row in filename.iterrows():
        if (row['Position']== ppos and row['Line up position'] == lpos):
            #print (row['PLAYER'],row['Line up position'],row['Position'],index)
            Playerindex.append(index)
    return Playerindex

def returnstartindex (ppos,filename):
    Playerindex = []
    for index,row in filename.iterrows():
        if (row['Line up position']== ppos):
            Playerindex.append(index)
    return Playerindex
    
    
         

if __name__ == '__main__':
    
    linearPredict = []
    LogisticPredict = []

    #cofe = []
    #lcofe = []

    testseasons = {'2012-13','2014-15','2016-17'}
    trainseasons = {'2013-14','2015-16'}
    #seasons = {'2016-17'}
    #years = {'2013','2014','2015','2016','2017'}
    foldername = 'predicte player/'
    subfolder = 'perdef/'
    traindata = pd.DataFrame()
    testdata = pd.DataFrame()
    for i in trainseasons:
        df = pd.read_csv(foldername+subfolder+i+'TeamTotalPERDEF_Train.csv')
        traindata = pd.concat([traindata,df])
    for i in testseasons:
        df2 = pd.read_csv(foldername+subfolder+i+'TeamTotalPERDEF_Train.csv')
    #df3 = pd.read_csv(foldername+subfolder+'2015-16'+'TeamTotalPERDEFForTrain.csv')
        testdata =pd.concat([testdata,df2])
    #testdata =pd.concat([testdata,df3])
    #traindata.to_csv(foldername+subfolder+'train.csv')
    #testdata.to_csv(foldername+subfolder+'test.csv')
    w1,w2,w3,w4,w0,linearPredict= LogisticRegressionTeam(traindata,testdata)
    '''
    lw1,lw2,lw3,lw4,lw0,LogisticPredict= LogisticRegressionTeam(traindata,testdata)
    #print(coef[0],coef[1])
    originalWinrate = testdata[['TEAM','WIN%']]
    print(originalWinrate)
    print(linearPredict)
    print(len(originalWinrate[['TEAM']]),len(linearPredict))
    originalWinrate['Predicted win-linear']= linearPredict
    originalWinrate['Predicted win-logistic']= LogisticPredict
    #print(originalWinrate)
    originalWinrate.to_csv(foldername+ subfolder + '12-13_14-15_16-17predictionPERDEF.csv')

    
    result=open(foldername + subfolder + "PER_DEFLinearAndLogicticRregressionWeight","w")

    result.write("weights are for the parameters in the below order:")
    result.write("\n\nLinear Weight:")
    result.write("\n\nw1 : TotalB_DEF:%f , w2 : TotalB_PER:%f , w3 :TotalS_DEF:%f , w4 :TotalS_PER:%f , w0 :%f" %(w1,w2,w3,w4,w0))
    result.write("\n\nLogistic Weight:")
    result.write("\n\nw1 : TotalB_DEF:%f , w2 : TotalB_PER:%f , w3 :TotalS_DEF:%f , w4 :TotalS_PER:%f , w0 :%f" %(lw1,lw2,lw3,lw4,lw0))

    result.close()
    '''
    #coef = np.array(coef)
    
    
    lineUpPositions=['S','B']
    positionsCategory=['C','PF','PG','SF','SG']
    PlayersStatsfile = pd.read_csv(foldername+subfolder+'2016-17testfile.csv')
    playerlist = []
    playersalary = []
    playerteam = []
    Time = []
    PER = []    
    DEF = []
    lineUpPositions =  []
    positions =  []
    for index, row in PlayersStatsfile.iterrows():
        playerlist.append(row['PLAYER'])
        playersalary.append(row['Salary'])
        playerteam.append(row['TEAM'])
        Time.append(row['MIN'])
        PER.append(row['PER'])
        DEF.append(row['DEF'])
        lineUpPositions.append(row['Line up position'])
        positions.append(row['Position'])
        
    #print(playerteam)

    #playerlist = PlayersStatsfile[['PLAYER']]
    prob = pulp.LpProblem("playersrecommendStrategy4lin", pulp.LpMinimize)
    isIncluded=pulp.LpVariable.dicts("isIncluded",indexs=playerlist,lowBound=0, upBound=1, cat='Integer')
    #print(isIncluded)
    #PlayersStatsfile = pd.read_csv(foldername+'')

    '''
    playerlist = PlayersStatsfile[['PLAYER']]
    playersalary = PlayersStatsfile[['Salary']]
    playerteam = PlayersStatsfile[['TEAM']]
    FGM =PlayersStatsfile[['FGM']]
    PTS =PlayersStatsfile[['PTS']]
    FGA =PlayersStatsfile[['FGA']]
    _3PM =PlayersStatsfile[['3PM']]
    _3PA =PlayersStatsfile[['3PA']]
    FTM =PlayersStatsfile[['FTM']]
    FTA =PlayersStatsfile[['FTA']]
    OREB =PlayersStatsfile[['OREB']]
    DREB =PlayersStatsfile[['DREB']]
    AST =PlayersStatsfile[['AST']]
    TOV =PlayersStatsfile[['TOV']]
    STL =PlayersStatsfile[['STL']]
    BLK =PlayersStatsfile[['BLK']]
    PF =PlayersStatsfile[['PF']]
    DEF =PlayersStatsfile[['DEF']]
    lineUpPositions = PlayersStatsfile[['Line up position']]
    positions = PlayersStatsfile[['Position']]
    '''


    #starterindex = returnstartindex('S',PlayersStatsfile)
    #backupindex = returnstartindex('B',PlayersStatsfile)
    
    #print(starterindex)
    #print(backupindex)

    for lpos in lineUpPositions:
        for ppos in positionsCategory:
             playerindex = returnindex(lpos,ppos,PlayersStatsfile)
             #print (playerindex)
             prob+=lpSum(isIncluded[playerlist[index]] for index in playerindex)==1

    starterindex = returnstartindex('S',PlayersStatsfile)
    backupindex = returnstartindex('B',PlayersStatsfile)
   

    prob+=lpSum(isIncluded[playerlist[index]]*playersalary[index] for index in range(len(playerlist)))#<=94140000
    #prob+=lpSum(isIncluded[playerlist[index]]*Time[index] for index in range(len(playerlist)))<=240

    prob+=(w1*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in backupindex))+(w2*lpSum(isIncluded[playerlist[index]]*PER[index] for index in backupindex))+(w3*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in starterindex))+(w4*lpSum(isIncluded[playerlist[index]]*PER[index] for index in starterindex))+w0>=1.496
    status = prob.solve()

    #print "RECOMMENDED PLAYERS FOR MAX OBJECTIVE:"
    #perresult.write("RECOMMENDED PLAYERS FOR MAX OBJECTIVE:")
    totalSalaryCost_max=0
    TotalTime = 0
    S_DEF = []
    S_PER = []
    B_DEF = []
    B_PER = []
    for i, player in enumerate(playerlist):
        if(value(isIncluded[player]) == 1.0):
            TotalTime+= Time[i]
            totalSalaryCost_max += playersalary[i]
            print("Name:", player, ", LineupPosition:",
                  lineUpPositions[i], ", Position:", positions[i],", MIN:", Time[i], ", Salary:", playersalary[i], ", Current Team:", playerteam[i])
            if (lineUpPositions[i]=='S'):
                S_DEF.append(DEF[i])
                S_PER.append(PER[i])
            else:
                B_DEF.append(DEF[i])
                B_PER.append(PER[i])
                


    
    #print(sum(S_3PA),w16)        #perresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))
    winrate = w1*sum(B_DEF)+w2*sum(B_PER)+w3*sum(S_DEF)+w4*sum(S_PER)+w0
    #prob+=(w1*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in backupindex))+(w2*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in backupindex))+(w3*lpSum(isIncluded[playerlist[index]]*AST[index] for index in backupindex))+(w4*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in backupindex))+(w5*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in backupindex))+(w6*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in backupindex))+(w7*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in backupindex))+(w8*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in backupindex))+(w9*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in backupindex))+(w10*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in backupindex)) +(w11*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in backupindex))+(w12*lpSum(isIncluded[playerlist[index]]*PF[index] for index in backupindex))+(w13*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in backupindex)) +(w14*lpSum(isIncluded[playerlist[index]]*STL[index] for index in backupindex)) +(w15*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in backupindex)) +(w16*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in starterindex))+(w17*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in starterindex))+(w18*lpSum(isIncluded[playerlist[index]]*AST[index] for index in starterindex))+(w19*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in starterindex))+(w20*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in starterindex)) +(w21*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in starterindex))+(w22*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in starterindex))+(w23*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in starterindex))+(w24*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in starterindex)) +(w25*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in starterindex))+(w26*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in starterindex))+(w27*lpSum(isIncluded[playerlist[index]]*PF[index] for index in starterindex))+(w28*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in starterindex))+(w29*lpSum(isIncluded[playerlist[index]]*STL[index] for index in starterindex))+(w30*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in starterindex))+w0
    #status = prob.solve()
    print ("Total salary is :",totalSalaryCost_max)
    print ('Team winning rate is :',1/(1+(math.e**(-winrate))))
    #print('Team winning rate is :',winrate)
    print('Team total MIN is:', TotalTime)
    print("\nMinimum Objective function value:",pulp.value(prob.objective))