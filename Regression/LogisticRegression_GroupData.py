import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import pandas as pd 


#season = '2017'
#filename = 'gamesPerDef'+ season +'.csv'
file = pd.read_csv('nba_elo.csv')
testfile = pd.read_csv('teams6.csv')
preictions = []

testX = testfile[[
      'HometeamElo',
      'VisitTeamElo',
      'Home_OREB',
      'Home_DREB',
      'Home_REB',
      'Home_AST',
      'Home_STL',
      'Home_BLK',
      'Home_TOV',
      'Home_PF',
      'Home_PTS',
      'Home_last20_WL',
      'Home_last15_WL',
      'Home_last10_WL',
      'Home_last5_WL',
      'Visit_OREB',
      'Visit_DREB',
      'Visit_REB',
      'Visit_AST',
      'Visit_STL',
      'Visit_BLK',
      'Visit_TOV',
      'Visit_PF',
      'Visit_PTS',
      'Visit_last20_WL',
      'Visit_last15_WL',
      'Visit_last10_WL',
      'Visit_last5_WL',
      ]]
      

X = file[[
      'HometeamElo',
      'VisitTeamElo',
      'Home_OREB',
      'Home_DREB',
      'Home_REB',
      'Home_AST',
      'Home_STL',
      'Home_BLK',
      'Home_TOV',
      'Home_PF',
      'Home_PTS',
      'Home_last20_WL',
      'Home_last15_WL',
      'Home_last10_WL',
      'Home_last5_WL',
      'Visit_OREB',
      'Visit_DREB',
      'Visit_REB',
      'Visit_AST',
      'Visit_STL',
      'Visit_BLK',
      'Visit_TOV',
      'Visit_PF',
      'Visit_PTS',
      'Visit_last20_WL',
      'Visit_last15_WL',
      'Visit_last10_WL',
      'Visit_last5_WL',
      ]]
Y = file['HomeTeamWL']
testY = testfile['HomeTeamWL']

logreg = linear_model.LogisticRegression()

logreg.fit(X,Y)
preictions = logreg.predict(X)
print (logreg.coef_, " ", logreg.intercept_)
print ("\n",preictions)
print(logreg.score(testX,testY))
#print()
'''
for i,row in testfile.iterrows():
    if (row['WoL'] == )
'''
'''
confusion_matrix = confusion_matrix(Y,preictions)
print (confusion_matrix)
'''
