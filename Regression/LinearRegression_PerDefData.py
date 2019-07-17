import pandas as pd 
import numpy as np 
from pulp import*
from sklearn import linear_model
season = '2017'
filename = 'gamesPerDef'+ season +'.csv'
testfile = pd.read_csv(filename)
predictrate = []
predictWoL = 0
count = 0
totalgame = 0


def linearRegressionPerDef(filename):
    df = pd.read_csv(filename)
    X=df[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer','VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
    #df['one']=1
    Y=df['WoL']
    lm=linear_model.LinearRegression()
    model=lm.fit(X,Y)
    w1=lm.coef_[0]
    w2=lm.coef_[1]
    w3=lm.coef_[2]
    w4=lm.coef_[3]
    w5=lm.coef_[4]
    w6=lm.coef_[5]
    w7=lm.coef_[6]
    w8=lm.coef_[7]
    #w9=lm.coef_[8]
    #w8=lm.coef_[8]
    return w1,w2,w3,w4,w5,w6,w7,w8,lm.intercept_

perresult=open("pernewdefstrategylinear","w")#打开文件，‘w’为写入模式
#defining constants

(w1,w2,w3,w4,w5,w6,w7,w8,intercept)=linearRegressionPerDef(filename)
perresult.write("Model parameters are:") 
perresult.write("\n w1: %f, w2: %f, w3: %f, w4: %f, w5: %f, w6: %f, w7: %f, w8: %f, intercept: %f" %(w1,w2,w3,w4,w5,w6,w7,w8,intercept))
print ("printing coefficients")
print ("\n w1=",w1," w2=",w2," w3=",w3," w4=",w4," w5=",w5," w6=",w6," w7=",w7," w8=",w8," intercept=",intercept)

for index,row in testfile.iterrows():
    totalgame+= 1
    

    if((row['HomeTeamBackupDef']*w1 + row['HomeTeambackupPer']*w2 + row['HomeTeamstartupDef']*w3 + row['HomeTeamstartupPer']*w4 + row['VisitBackupDef']*w5 + row['VisitTeambackupPer']*w6 + row['VisitTeamstartupDef']*w7 + row['VisitTeamstartupPer']*w8  + intercept)>1):
        predictWoL = 1
        predictrate.append(predictWoL)
    else:
        predictWoL = 0
        predictrate.append(predictWoL)
    
    print ("rate:",(row['HomeTeamBackupDef']*w1 + row['HomeTeambackupPer']*w2 + row['HomeTeamstartupDef']*w3 + row['HomeTeamstartupPer']*w4 + row['VisitBackupDef']*w5 + row['VisitTeambackupPer']*w6 + row['VisitTeamstartupDef']*w7 + row['VisitTeamstartupPer']*w8  + intercept) ," predictWoL: ",predictWoL," WoL: ",row['WoL'])
    if (predictWoL == row['WoL']):
        count+= 1

predictWoLrate = count/totalgame
print ("\n the Accuracy of Prediction is: ")
print (predictWoLrate)
perresult.close()