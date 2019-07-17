import pandas as pd 
import numpy as np 
import math

from pulp import*
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
season = '2017'
filename = 'gamesPerDef'+ season +'.csv'
testfile = pd.read_csv(filename)
predictrate = []
predictWoL = 0
count = 0
totalgame = 0


def logisticRegressionPerDef(filename):
    df = pd.read_csv(filename)
    x_test = df[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer','VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
    X=df[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer','VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
    X['ones']=1
    Y=df['WoL']
    x_test['intercept'] = 1
    mod = sm.Logit(Y,X)
    res = mod.fit()
    #map(int,x_test)
    (w1,w2,w3,w4,w5,w6,w7,w8,intercept)= res.params
    x_test['predictions'] = res.predict(x_test)
    return w1,w2,w3,w4,w5,w6,w7,w8,intercept,x_test['predictions']

perresult=open("pernewdefstrategylinear","w")#打开文件，‘w’为写入模式
#defining constants

(w1,w2,w3,w4,w5,w6,w7,w8,intercept,prediction)=logisticRegressionPerDef(filename)


confusion_matrix = confusion_matrix(testfile['WoL'],prediction)

perresult.write("Model parameters are:") 
perresult.write("\n w1: %f, w2: %f, w3: %f, w4: %f, w5: %f, w6: %f, w7: %f, w8: %f intercept: %f" %(w1,w2,w3,w4,w5,w6,w7,w8,intercept))
print ("printing coefficients")
print ("\n w1=",w1," w2=",w2," w3=",w3," w4=",w4," w5=",w5," w6=",w6," w7=",w7," w8=",w8," intercept=",intercept)
print ("\n",prediction)
'''
print (confusion_matrix)

for index,row in testfile.iterrows():
    totalgame+= 1
    y = row['HomeTeamBackupDef']*w1 + row['HomeTeambackupPer']*w2 + row['HomeTeamstartupDef']*w3 + row['HomeTeamstartupPer']*w4 + row['VisitBackupDef']*w5 + row['VisitTeambackupPer']*w6 + row['VisitTeamstartupDef']*w7 + row['VisitTeamstartupPer']*w8 + intercept
    target = (1/(1 + math.exp(-y)))/2
    if(target > 1/2):
        predictWoL = 1
        predictrate.append(predictWoL)
    else:
        predictWoL = 0
        predictrate.append(predictWoL)
    
    print ("rate:",target ," predictWoL: ",predictWoL," WoL: ",row['WoL'])
    if (predictWoL == row['WoL']):
        count+= 1

predictWoLrate = count/totalgame
print ("\n the Accuracy of Prediction is: ")
print (predictWoLrate)
'''
