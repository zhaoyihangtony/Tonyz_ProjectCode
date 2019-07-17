from sklearn import linear_model
from sklearn.linear_model.ridge import Ridge
import math
from pulp import *
import pandas as pd
import numpy as np  
import statsmodels.api as sm

def LinearRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['TotalB_3PA',
          'TotalB_3PM',
          'TotalB_AST',
          'TotalB_BLK',
          'TotalB_DEF',
          'TotalB_DREB',
          'TotalB_FGA',
          'TotalB_FGM',
          'TotalB_FTA',
          'TotalB_FTM',
          'TotalB_OREB',
          'TotalB_PF',
          'TotalB_STL',
          'TotalB_TOV',
          'TotalS_3PA',
          'TotalS_3PM',
          'TotalS_AST',
          'TotalS_BLK',
          'TotalS_DEF',
          'TotalS_DREB',
          'TotalS_FGA',
          'TotalS_FGM',
          'TotalS_FTA',
          'TotalS_FTM',
          'TotalS_OREB',
          'TotalS_PF',
          'TotalS_STL',
          'TotalS_TOV']]
    testX=testfile[['TotalB_3PA',
          'TotalB_3PM',
          'TotalB_AST',
          'TotalB_BLK',
          'TotalB_DEF',
          'TotalB_DREB',
          'TotalB_FGA',
          'TotalB_FGM',
          'TotalB_FTA',
          'TotalB_FTM',
          'TotalB_OREB',
          'TotalB_PF',
          'TotalB_STL',
          'TotalB_TOV',
          'TotalS_3PA',
          'TotalS_3PM',
          'TotalS_AST',
          'TotalS_BLK',
          'TotalS_DEF',
          'TotalS_DREB',
          'TotalS_FGA',
          'TotalS_FGM',
          'TotalS_FTA',
          'TotalS_FTM',
          'TotalS_OREB',
          'TotalS_PF',
          'TotalS_STL',
          'TotalS_TOV']]
    Y=filename['WIN%']
    testY=testfile['WIN%']
    lm = linear_model.LinearRegression()
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
    w15 = lm.coef_[14]
    w16 = lm.coef_[15]
    w17 = lm.coef_[16]
    w18 = lm.coef_[17]
    w19 = lm.coef_[18]
    w20 = lm.coef_[19]
    w21 = lm.coef_[20]
    w22 = lm.coef_[21]
    w23 = lm.coef_[22]
    w24 = lm.coef_[23]
    w25 = lm.coef_[24]
    w26 = lm.coef_[25]
    w27 = lm.coef_[26]
    w28 = lm.coef_[27]
    #print(lm.coef_)


    #(w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29)=lm.coef_
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,lm.intercept_,linearmodelpredict

def LogisticRegressionTeam(filename,testfile):
    #df = pd.read_csv(filename)
    X=filename[['TotalB_3PA',
          'TotalB_3PM',
          'TotalB_AST',
          'TotalB_BLK',
          'TotalB_DEF',
          'TotalB_DREB',
          'TotalB_FGA',
          'TotalB_FGM',
          'TotalB_FTA',
          'TotalB_FTM',
          'TotalB_OREB',
          'TotalB_PF',
          'TotalB_STL',
          'TotalB_TOV',
          'TotalS_3PA',
          'TotalS_3PM',
          'TotalS_AST',
          'TotalS_BLK',
          'TotalS_DEF',
          'TotalS_DREB',
          'TotalS_FGA',
          'TotalS_FGM',
          'TotalS_FTA',
          'TotalS_FTM',
          'TotalS_OREB',
          'TotalS_PF',
          'TotalS_STL',
          'TotalS_TOV']]
    testX=testfile[['TotalB_3PA',
          'TotalB_3PM',
          'TotalB_AST',
          'TotalB_BLK',
          'TotalB_DEF',
          'TotalB_DREB',
          'TotalB_FGA',
          'TotalB_FGM',
          'TotalB_FTA',
          'TotalB_FTM',
          'TotalB_OREB',
          'TotalB_PF',
          'TotalB_STL',
          'TotalB_TOV',
          'TotalS_3PA',
          'TotalS_3PM',
          'TotalS_AST',
          'TotalS_BLK',
          'TotalS_DEF',
          'TotalS_DREB',
          'TotalS_FGA',
          'TotalS_FGM',
          'TotalS_FTA',
          'TotalS_FTM',
          'TotalS_OREB',
          'TotalS_PF',
          'TotalS_STL',
          'TotalS_TOV']]
    Y=filename[['WIN%']]
    testY = testfile[['WIN%']]
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
    w15 = res.params[14]
    w16 = res.params[15]
    w17 = res.params[16]
    w18 = res.params[17]
    w19 = res.params[18]
    w20 = res.params[19]
    w21 = res.params[20]
    w22 = res.params[21]
    w23 = res.params[22]
    w24 = res.params[23]
    w25 = res.params[24]
    w26 = res.params[25]
    w27 = res.params[26]
    w28 = res.params[27]
    w0 = res.params[28]
    #print(res.params)
    logisticmodelpredict = res.predict(testX)
    
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w0,logisticmodelpredict


def returnindex (lpos,ppos,filename):
    Playerindex = []
    for index,row in filename.iterrows():
        if (row['Line up position'] == lpos and row['Position']== ppos):
            #print (row['PLAYER'],row['Line up position'],row['Line up position'],index)
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
    subfolder = '14attrs/'
    traindata = pd.DataFrame()
    testdata = pd.DataFrame()
    for i in trainseasons:
        df = pd.read_csv(foldername+'TrainData/'+i+'TeamTotalStatsForTrain.csv')
        traindata = pd.concat([traindata,df])
    for i in testseasons:
        df2 = pd.read_csv(foldername+'TrainData/'+i+'TeamTotalStatsForTrain.csv')
        testdata =pd.concat([testdata,df2])
    #traindata.to_csv(foldername+subfolder+'train.csv')
    #testdata.to_csv(foldername+subfolder+'test.csv')
    w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w0,linearPredict= LogisticRegressionTeam(traindata,testdata)
    
    #result.o
    #coef = np.array(coef)
    #lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8,lw9,lw10,lw11,lw12,lw13,lw14,lw15,lw16,lw17,lw18,lw19,lw20,lw21,lw22,lw23,lw24,lw25,lw26,lw27,lw28,lw0,LogisticPredict= LogisticRegressionTeam(traindata,testdata)
    '''
    originalWinrate = testdata[['TEAM','WIN%']]
    print(originalWinrate)
    print(linearPredict)
    print(len(originalWinrate[['TEAM']]),len(linearPredict))
    originalWinrate['Predicted win-linear']= linearPredict
    originalWinrate['Predicted win-logistic']= LogisticPredict
    #print(originalWinrate)
    originalWinrate.to_csv(foldername+ subfolder + '13-14_15-16predictionLinearLogistic14attrs.csv')

    
    result=open(foldername + subfolder + "14AttrsLinearLogistic_Weight","w")

    result.write("weights are for the parameters in the below order:")
    result.write("\n\nLinear Weight:")
    result.write("\n\nw1 : TotalB_3PA:%f , w2 : TotalB_3PM:%f , w3 :TotalB_AST:%f , w4 :TotalB_BLK:%f , w5 :TotalB_DEF:%f ,\n w6 :TotalB_DREB:%f , w7 :TotalB_FGA:%f , w8 :TotalB_FGM:%f , w9 :TotalB_FTA:%f , w10 :TotalB_FTM:%f ,\n w11 :TotalB_OREB:%f , w12 :TotalB_PF:%f , w13 :TotalB_STL:%f , w14 :TotalB_TOV:%f ,\n w15 :TotalS_3PA:%f , w16 :TotalS_3PM:%f , w17 :TotalS_AST:%f , w18 :TotalS_BLK:%f , w19 :TotalS_DEF:%f ,\n w20 :TotalS_DREB:%f , w21 :TotalS_FGA:%f , w22 :TotalS_FGM:%f , w23 :TotalS_FTA:%f , w24 :TotalS_FTM:%f ,\n w25 :TotalS_OREB:%f , w26 :TotalS_PF:%f , w27 :TotalS_STL:%f  ,w28 :TotalS_TOV:%f , w0 :%f" %(w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w0))
    result.write("\n\nLogistic Weight:")
    result.write("\n\nw1 : TotalB_3PA:%f , w2 : TotalB_3PM:%f , w3 :TotalB_AST:%f , w4 :TotalB_BLK:%f , w5 :TotalB_DEF:%f ,\n w6 :TotalB_DREB:%f , w7 :TotalB_FGA:%f , w8 :TotalB_FGM:%f , w9 :TotalB_FTA:%f , w10 :TotalB_FTM:%f ,\n w11 :TotalB_OREB:%f , w12 :TotalB_PF:%f , w13 :TotalB_STL:%f , w14 :TotalB_TOV:%f ,\n w15 :TotalS_3PA:%f , w16 :TotalS_3PM:%f , w17 :TotalS_AST:%f , w18 :TotalS_BLK:%f , w19 :TotalS_DEF:%f ,\n w20 :TotalS_DREB:%f , w21 :TotalS_FGA:%f , w22 :TotalS_FGM:%f , w23 :TotalS_FTA:%f , w24 :TotalS_FTM:%f ,\n w25 :TotalS_OREB:%f , w26 :TotalS_PF:%f , w27 :TotalS_STL:%f , w28 :TotalS_TOV:%f , w0 :%f" %(lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8,lw9,lw10,lw11,lw12,lw13,lw14,lw15,lw16,lw17,lw18,lw19,lw20,lw21,lw22,lw23,lw24,lw25,lw26,lw27,lw28,lw0))

    result.close()
   '''
    
    
    #print(coef[0],coef[1])
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
    prob = pulp.LpProblem("playersrecommendStrategy4lin", pulp.LpMaximize)
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

    #playerindex = []
    #starterlist = []
    #backuplist = []
    starterindex = returnstartindex('S',PlayersStatsfile)
    backupindex = returnstartindex('B',PlayersStatsfile)
    #print(starterindex)
    #print(backupindex)

    for lpos in lineUpPositions:
        for ppos in positionsCategory:
            playerindex = returnindex(lpos,ppos,PlayersStatsfile)
           # print (playerindex)
            prob+=lpSum(isIncluded[playerlist[index]] for index in playerindex)==1

    #starterindex = returnstartindex('S',PlayersStatsfile)
    #backupindex = returnstartindex('B',PlayersStatsfile)
   

    prob+=lpSum(isIncluded[playerlist[index]]*playersalary[index] for index in range(len(playerlist)))<=94140000

    prob+=(w1*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in backupindex))+(w2*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in backupindex))+(w3*lpSum(isIncluded[playerlist[index]]*AST[index] for index in backupindex))+(w4*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in backupindex))+(w5*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in backupindex))+(w6*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in backupindex))+(w7*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in backupindex))+(w8*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in backupindex))+(w9*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in backupindex))+(w10*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in backupindex)) +(w11*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in backupindex))+(w12*lpSum(isIncluded[playerlist[index]]*PF[index] for index in backupindex))+(w13*lpSum(isIncluded[playerlist[index]]*STL[index] for index in backupindex)) +(w14*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in backupindex)) +(w15*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in starterindex))+(w16*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in starterindex))+(w17*lpSum(isIncluded[playerlist[index]]*AST[index] for index in starterindex))+(w18*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in starterindex))+(w19*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in starterindex)) +(w20*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in starterindex))+(w21*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in starterindex))+(w22*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in starterindex))+(w23*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in starterindex)) +(w24*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in starterindex))+(w25*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in starterindex))+(w26*lpSum(isIncluded[playerlist[index]]*PF[index] for index in starterindex))+(w27*lpSum(isIncluded[playerlist[index]]*STL[index] for index in starterindex))+(w28*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in starterindex))+w0#>=1.496
    status = prob.solve()

    #print "RECOMMENDED PLAYERS FOR MAX OBJECTIVE:"
    #perresult.write("RECOMMENDED PLAYERS FOR MAX OBJECTIVE:")
    totalSalaryCost_max=0
    B_FGM = []
    #B_PTS = []
    B_FGA = []
    B_3PM = []
    B_3PA = []
    B_FTM = []
    B_FTA = []
    B_OREB = []
    B_DREB = []
    B_AST = []
    B_TOV = []
    B_STL = []
    B_BLK = []
    B_PF = []
    B_DEF = []
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
            totalSalaryCost_max += playersalary[i]
            print("Name:", player, ", LineupPosition:",
                  lineUpPositions[i], ", Position:", positions[i], ", Salary:", playersalary[i], ", Current Team:", playerteam[i])
            if (lineUpPositions[i]=='S'):
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
            else:
                B_3PA.append(_3PA[i])
                B_FGM.append(FGM[i])
                #B_PTS.append(PTS[i])
                B_FGA.append(FGA[i])
                B_3PM.append(_3PM[i])
                B_FTM.append(FTM[i])
                B_FTA.append(FTA[i])
                B_OREB.append(OREB[i])
                B_DREB.append(DREB[i])
                B_AST.append(AST[i])
                B_TOV.append(TOV[i])
                B_STL.append(STL[i])
                B_BLK.append(BLK[i])
                B_PF.append(PF[i])
                B_DEF.append(DEF[i])
                
                #S_3PA.append(_3PA[i])
            
    
    #print(sum(S_3PA),w16)        #perresult.write("\n Name: %s, LineupPosition: %s, Position: %s, PER: %f, Salary: %f, Defensive rating: %f, Current Team: %s" %(player,playersLineUpPosition[i],playersPosition[i],playersPER[i],playersSalary[i],playersDefenseRating[i],teamName[i]))
    winrate = w1*sum(B_3PA)+w2*sum(B_3PM)+w3*sum(B_AST)+w4*sum(B_BLK)+w5*sum(B_DEF)+w6*sum(B_DREB)+w7*sum(B_FGA)+w8*sum(B_FGM)+w9*sum(B_FTA)+w10*sum(B_FTM)+w11*sum(B_OREB)+w12*sum(B_PF)+w13*sum(B_STL)+w14*sum(B_TOV)+w15*sum(S_3PA)+w16*sum(S_3PM)+w17*sum(S_AST)+w18*sum(S_BLK)+w19*sum(S_DEF)+w20*sum(S_DREB)+w21*sum(S_FGA)+w22*sum(S_FGM)+w23*sum(S_FTA)+w24*sum(S_FTM)+w25*sum(S_OREB)+w26*sum(S_PF)+w27*sum(S_STL)+w28*sum(S_TOV)+w0
    #prob+=(w1*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in backupindex))+(w2*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in backupindex))+(w3*lpSum(isIncluded[playerlist[index]]*AST[index] for index in backupindex))+(w4*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in backupindex))+(w5*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in backupindex))+(w6*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in backupindex))+(w7*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in backupindex))+(w8*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in backupindex))+(w9*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in backupindex))+(w10*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in backupindex)) +(w11*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in backupindex))+(w12*lpSum(isIncluded[playerlist[index]]*PF[index] for index in backupindex))+(w13*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in backupindex)) +(w14*lpSum(isIncluded[playerlist[index]]*STL[index] for index in backupindex)) +(w15*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in backupindex)) +(w16*lpSum(isIncluded[playerlist[index]]*_3PA[index] for index in starterindex))+(w17*lpSum(isIncluded[playerlist[index]]*_3PM[index] for index in starterindex))+(w18*lpSum(isIncluded[playerlist[index]]*AST[index] for index in starterindex))+(w19*lpSum(isIncluded[playerlist[index]]*BLK[index] for index in starterindex))+(w20*lpSum(isIncluded[playerlist[index]]*DEF[index] for index in starterindex)) +(w21*lpSum(isIncluded[playerlist[index]]*DREB[index] for index in starterindex))+(w22*lpSum(isIncluded[playerlist[index]]*FGA[index] for index in starterindex))+(w23*lpSum(isIncluded[playerlist[index]]*FGM[index] for index in starterindex))+(w24*lpSum(isIncluded[playerlist[index]]*FTA[index] for index in starterindex)) +(w25*lpSum(isIncluded[playerlist[index]]*FTM[index] for index in starterindex))+(w26*lpSum(isIncluded[playerlist[index]]*OREB[index] for index in starterindex))+(w27*lpSum(isIncluded[playerlist[index]]*PF[index] for index in starterindex))+(w28*lpSum(isIncluded[playerlist[index]]*PTS[index] for index in starterindex))+(w29*lpSum(isIncluded[playerlist[index]]*STL[index] for index in starterindex))+(w30*lpSum(isIncluded[playerlist[index]]*TOV[index] for index in starterindex))+w0
    #status = prob.solve()
    print ("Total salary is :",totalSalaryCost_max)
    print ('Team winning rate is :',1/(1+(math.e**(-winrate))))
    #print('Team winning rate is :',winrate)
    print("\nMinimum Objective function value:",pulp.value(prob.objective))