from sklearn import linear_model
import math
from pulp import *
import pandas as pd
import numpy as np  
import statsmodels.api as sm

def linearRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['Total_3PA',
          'Total_3PM',
          'Total_AST',
          'Total_BLK',
          'Total_DEF',
          'Total_DREB',
          'Total_FGA',
          'Total_FGM',
          'Total_FTA',
          'Total_FTM',
          'Total_OREB',
          'Total_PF',
          'Total_STL',
          'Total_TOV']]
    testX=testfile[['Total_3PA',
          'Total_3PM',
          'Total_AST',
          'Total_BLK',
          'Total_DEF',
          'Total_DREB',
          'Total_FGA',
          'Total_FGM',
          'Total_FTA',
          'Total_FTM',
          'Total_OREB',
          'Total_PF',
          'Total_STL',
          'Total_TOV']]
    Y=filename['WIN%']
    testY=testfile['WIN%']
    lm=linear_model.LinearRegression()
    model = lm.fit(X,Y)
    linearmodelpredict = lm.predict(testX)
    w1 = lm.coef_[0]
    w2 = lm.coef_[1]
    w3 = lm.coef_[2]
    w4 = lm.coef_[3]
    w5 = lm.coef_[4]
    w6 = lm.coef_[5]
    w7 = lm.coef_[6]
    w8 = lm.coef_[7]
    w9 = lm.coef_[8]
    w10 = lm.coef_[9]
    w11 = lm.coef_[10]
    w12 = lm.coef_[11]
    w13 = lm.coef_[12]
    w14 = lm.coef_[13]
    
    #print(lm.coef_)


    #(w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29)=lm.coef_
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,lm.intercept_,linearmodelpredict

def LogisticRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['Total_3PA',
          'Total_3PM',
          'Total_AST',
          'Total_BLK',
          'Total_DEF',
          'Total_DREB',
          'Total_FGA',
          'Total_FGM',
          'Total_FTA',
          'Total_FTM',
          'Total_OREB',
          'Total_PF',
          'Total_STL',
          'Total_TOV']]
    testX=testfile[['Total_3PA',
          'Total_3PM',
          'Total_AST',
          'Total_BLK',
          'Total_DEF',
          'Total_DREB',
          'Total_FGA',
          'Total_FGM',
          'Total_FTA',
          'Total_FTM',
          'Total_OREB',
          'Total_PF',
          'Total_STL',
          'Total_TOV']]
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
    w5 = res.params[4]
    w6 = res.params[5]
    w7 =res.params[6]
    w8 = res.params[7]
    w9 = res.params[8]
    w10 = res.params[9]
    w11 = res.params[10]
    w12 = res.params[11]
    w13 = res.params[12]
    w14 = res.params[13]
    w0 = res.params[14]
    #print(res.params)
    logisticmodelpredict = res.predict(testX)
    
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w0,logisticmodelpredict
def returnindex (ppos,filename):
    Playerindex = []
    for index,row in filename.iterrows():
        if (row['Position']== ppos):
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

    trainseasons = {'2012-13','2014-15','2016-17'}
    testseasons = {'2013-14','2015-16'}
    #seasons = {'2016-17'}
    #years = {'2013','2014','2015','2016','2017'}
    foldername = 'predicte player/'
    subfolder = 'TrainData/'
    traindata = pd.DataFrame()
    testdata = pd.DataFrame()
    for i in trainseasons:
        df = pd.read_csv(foldername+'Time/'+i+'TeamTotalStatsForTrain.csv')
        traindata = pd.concat([traindata,df])
    for i in testseasons:
        df2 = pd.read_csv(foldername+'Time/'+i+'TeamTotalStatsForTrain.csv')
        testdata =pd.concat([testdata,df2])
    #traindata.to_csv(foldername+subfolder+'train.csv')
    #testdata.to_csv(foldername+subfolder+'test.csv')
    w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w0,linearPredict= LogisticRegressionTeam(traindata,testdata)
    '''
    lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8,lw9,lw10,lw11,lw12,lw13,lw14,lw0,LogisticPredict= LogisticRegressionTeam(traindata,testdata)
    #print(coef[0],coef[1])
    originalWinrate = testdata[['TEAM','WIN%']]
    originalWinrate['Predicted win-linear']= linearPredict
    originalWinrate['Predicted win-logistic']= LogisticPredict
    #print(originalWinrate)
    originalWinrate.to_csv(foldername+ 'Time/'+ '13-14_15-16LinearLogisticPrediction_14attrs.csv')

    
    result=open(foldername + 'Time/' + "TimeConstrainLinearAndLogicticRregressionWeight_14sttrs","w")

    result.write("weights are for the parameters in the below order:")
    result.write("\n\nLinear Weight:")
    result.write("\n\nw1 : Total_3PA:%f , w2 : Total_3PM:%f , w3 :Total_AST:%f , w4 :Total_BLK:%f , w5 :Total_DEF:%f ,\n w6 :Total_DREB:%f , w7 :Total_FGA:%f , w8 :Total_FGM:%f , w9 :Total_FTA:%f , w10 :Total_FTM:%f ,\n w11 :Total_OREB:%f , w12 :Total_PF:%f , w13 :Total_STL:%f , w14 :Total_TOV:%f ,\n w0 :%f" %(w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w0))
    result.write("\n\nLogistic Weight:")
    result.write("\n\n\w1 : Total_3PA:%f , w2 : Total_3PM:%f , w3 :Total_AST:%f , w4 :Total_BLK:%f , w5 :Total_DEF:%f ,\n w6 :Total_DREB:%f , w7 :Total_FGA:%f , w8 :Total_FGM:%f , w9 :Total_FTA:%f , w10 :Total_FTM:%f ,\n w11 :Total_OREB:%f , w12 :Total_PF:%f , w13 :Total_STL:%f , w14 :Total_TOV:%f ,\n w0 :%f" %(lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8,lw9,lw10,lw11,lw12,lw13,lw14,lw0))

    result.close()
    #coef = np.array(coef)
    
    '''
    lineUpPositions=['S','B']
    positionsCategory=['C','PF','PG','SF','SG']
    PlayersStatsfile = pd.read_csv(foldername+'2016-17testfile.csv')
    playerlist = []
    playersalary = []
    playerteam = []
    Time = []
    FGM = []
    #PTS = []
    FGA = []
    _3PM = []
    _3PA = []
    FTM = []
    FTA = []
    OREB = []
    DREB = []
    AST = []
    TOV = []
    STL = []
    BLK = []
    PF = []
    DEF = []
    lineUpPositions =  []
    positions =  []
    for index, row in PlayersStatsfile.iterrows():
        playerlist.append(row['PLAYER'])
        playersalary.append(row['Salary'])
        playerteam.append(row['TEAM'])
        Time.append(row['MIN'])
        FGM.append(row['FGM'])
        #PTS.append(row['PTS'])
        FGA.append(row['FGA'])
        _3PM.append(row['3PM'])
        _3PA.append(row['3PA'])
        FTM.append(row['FTM'])
        FTA.append(row['FTA'])
        OREB.append(row['OREB'])
        DREB.append(row['DREB'])
        AST.append(row['AST'])
        TOV.append(row['TOV'])
        STL.append(row['STL'])
        BLK.append(row['BLK'])
        PF.append(row['PF'])
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

    #for lpos in lineUpPositions:
    for ppos in positionsCategory:
        playerindex = returnindex(ppos,PlayersStatsfile)
        #print (playerindex)
        prob+=lpSum(isIncluded[playerlist[index]] for index in playerindex)==2

    #starterindex = returnstartindex('S',PlayersStatsfile)
    #backupindex = returnstartindex('B',PlayersStatsfile)
   

    prob+=lpSum(isIncluded[playerlist[index]]*playersalary[index] for index in range(len(playerlist)))#<=94140000
    prob+=lpSum(isIncluded[playerlist[index]]*Time[index] for index in range(len(playerlist)))<=240

    prob+=(w1*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in range(len(playerlist))))+(w2*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in range(len(playerlist))))+(w3*lpSum(isIncluded[playerlist[index]]*AST[index] for index in range(len(playerlist))))+(w4*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in range(len(playerlist))))+(w5*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in range(len(playerlist))))+(w6*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in range(len(playerlist))))+(w7*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in range(len(playerlist))))+(w8*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in range(len(playerlist))))+(w9*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in range(len(playerlist))))+(w10*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in range(len(playerlist)))) +(w11*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in range(len(playerlist))))+(w12*lpSum(isIncluded[playerlist[index]]*PF[index] for index in range(len(playerlist)))) +(w13*lpSum(isIncluded[playerlist[index]]*STL[index] for index in range(len(playerlist)))) +(w14*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in range(len(playerlist))))+w0>=1.496
    status = prob.solve()

    #print "RECOMMENDED PLAYERS FOR MAX OBJECTIVE:"
    #perresult.write("RECOMMENDED PLAYERS FOR MAX OBJECTIVE:")
    totalSalaryCost_max=0
    TotalTime = 0
    S_FGM = []
    #S_PTS = []
    S_FGA = []
    S_3PM = []
    S_3PA = []
    S_FTM = []
    S_FTA = []
    S_OREB = []
    S_DREB = []
    S_AST = []
    S_TOV = []
    S_STL = []
    S_BLK = []
    S_PF = []
    S_DEF = []
    for i, player in enumerate(playerlist):
        if(value(isIncluded[player]) == 1.0):
            TotalTime+= Time[i]
            totalSalaryCost_max += playersalary[i]
            print("Name:", player, ", LineupPosition:",
                  lineUpPositions[i], ", Position:", positions[i],", MIN:", Time[i], ", Salary:", playersalary[i], ", Current Team:", playerteam[i])
            S_3PA.append(_3PA[i])
                #print (_3PA[i])
            S_FGM.append(FGM[i])
            #S_PTS.append(PTS[i])
            S_FGA.append(FGA[i])
            S_3PM.append(_3PM[i])
            S_FTM.append(FTM[i])
            S_FTA.append(FTA[i])
            S_OREB.append(OREB[i])
            S_DREB.append(DREB[i])
            S_AST.append(AST[i])
            S_TOV.append(TOV[i])
            S_STL.append(STL[i])
            S_BLK.append(BLK[i])
            S_PF.append(PF[i])
            S_DEF.append(DEF[i])

    
    #print(sum(S_3PA),w16)        #perresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))
    winrate = w1*sum(S_3PA)+w2*sum(S_3PM)+w3*sum(S_AST)+w4*sum(S_BLK)+w5*sum(S_DEF)+w6*sum(S_DREB)+w7*sum(S_FGA)+w8*sum(S_FGM)+w9*sum(S_FTA)+w10*sum(S_FTM)+w11*sum(S_OREB)+w12*sum(S_PF)+w13*sum(S_STL)+w14*sum(S_TOV)+w0
    #prob+=(w1*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in backupindex))+(w2*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in backupindex))+(w3*lpSum(isIncluded[playerlist[index]]*AST[index] for index in backupindex))+(w4*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in backupindex))+(w5*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in backupindex))+(w6*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in backupindex))+(w7*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in backupindex))+(w8*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in backupindex))+(w9*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in backupindex))+(w10*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in backupindex)) +(w11*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in backupindex))+(w12*lpSum(isIncluded[playerlist[index]]*PF[index] for index in backupindex))+(w13*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in backupindex)) +(w14*lpSum(isIncluded[playerlist[index]]*STL[index] for index in backupindex)) +(w15*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in backupindex)) +(w16*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in starterindex))+(w17*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in starterindex))+(w18*lpSum(isIncluded[playerlist[index]]*AST[index] for index in starterindex))+(w19*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in starterindex))+(w20*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in starterindex)) +(w21*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in starterindex))+(w22*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in starterindex))+(w23*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in starterindex))+(w24*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in starterindex)) +(w25*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in starterindex))+(w26*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in starterindex))+(w27*lpSum(isIncluded[playerlist[index]]*PF[index] for index in starterindex))+(w28*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in starterindex))+(w29*lpSum(isIncluded[playerlist[index]]*STL[index] for index in starterindex))+(w30*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in starterindex))+w0
    #status = prob.solve()
    print ("Total salary is :",totalSalaryCost_max)
    print ('Team winning rate is :',1/(1+(math.e**(-winrate))))
    #print('Team winning rate is :',winrate)
    print('Team total MIN is:', TotalTime)
    print("\nMinimum Objective function value:",pulp.value(prob.objective))


        
            
            