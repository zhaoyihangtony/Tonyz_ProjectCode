import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd 

'''
trainseason = '2015'
filename = 'gamesPerDef'+ trainseason +'.csv'
file2015 = pd.read_csv(filename)
file2014 = pd.read_csv('gamesPerDef2014.csv')
file2013 = pd.read_csv('gamesPerDef2013.csv')
file2012 = pd.read_csv('gamesPerDef2012.csv')
'''
file2016 = pd.read_csv('gamesPerDef2016.csv')

testseason = '2017'
testfilename = 'gamesPerDef'+ testseason +'.csv'
testfile = pd.read_csv(testfilename)
file = pd.concat([file2012,file2013,file2014,file2015,file2016])
preictions = []
#print(file)


trainX = file2016[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer','VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
trainY = file2016['WoL']
testX = testfile[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer','VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
testY = testfile['WoL']
##plt.scatter(X[:,0],Y[:,0],marker='o',c=Y)
#bdt = AdaBoostClassifier(base_estimator=linear_model.LogisticRegression(),algorithm='SAMME',n_estimators=2000,learning_rate=0.02)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5),algorithm='SAMME.R',n_estimators=500,learning_rate=0.2)

bdt.fit(trainX,trainY)
preictions = bdt.predict(testX)
print(bdt.estimator_weights_)
print("score : ",bdt.score(testX,testY))
print ("\n",preictions)
'''
for i,row in testfile.iterrows():
    if (row['WoL'] == )
'''
confusion_matrix = confusion_matrix(testY,preictions)
print (confusion_matrix)
