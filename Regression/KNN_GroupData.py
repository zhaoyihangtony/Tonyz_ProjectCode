from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import BaggingClassifier
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 

file2015 = pd.read_csv('gamesPerDef2015.csv')
file2014 = pd.read_csv('gamesPerDef2014.csv')
file2013 = pd.read_csv('gamesPerDef2013.csv')
file2012 = pd.read_csv('gamesPerDef2012.csv')
file2016 = pd.read_csv('gamesPerDef2016.csv')

testseason = '2017'
testfilename = 'gamesPerDef'+ testseason +'.csv'
testfile = pd.read_csv(testfilename)
file = pd.concat([file2012,file2013,file2014,file2015,file2016])
trainX = file[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer',
    'VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
trainY = file['WoL']
testX = testfile[['HomeTeamBackupDef','HomeTeambackupPer','HomeTeamstartupDef','HomeTeamstartupPer',
    'VisitBackupDef','VisitTeambackupPer','VisitTeamstartupDef','VisitTeamstartupPer']]
testY = testfile['WoL']

knn = KNeighborsClassifier(n_neighbors = 150,weights='distance',algorithm='brute',p=1)
knn.fit(trainX,trainY)
pred = knn.predict(testX)
confusion_matrix = confusion_matrix(testY,pred)
rate= knn.score(testX,testY)
print('\n',confusion_matrix)
print ('\n',rate)

'''
bdt = BaggingClassifier(KNeighborsClassifier(n_neighbors=15),max_samples=0.5, max_features=0.5)

bdt.fit(Normalized_trainX,trainY)
preictions = bdt.predict(Normalized_testX)
#print(bdt.estimator_weights_)
print("score : ",bdt.score(Normalized_testX,testY))

'''
#print (b)
#print(file)
'''
import numpy as np 
a= [1,2,3,4,5]
b= [1,2,3,4,5]
c= [1,2,3,4,5]
d= [1,2,3,4,5]
score = np.array([a,b,c,d])
#

a = score.reshape((-1,5))
b= MinMaxScaler().fit_transform(a)
print (b)
'''