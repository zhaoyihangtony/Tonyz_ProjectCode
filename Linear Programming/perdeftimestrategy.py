from pulp import *
from sklearn import linear_model
import pandas as pd
from functions import *

teamName = []
playersName = []
playersLineUpPosition = []
playersPosition = []
playersPER = []
playersSalary = []
playersDefenseRating = []

perresult = open("pernewdeftimestrategylinear", "w")

def linearregression ():
    filename = ''
    file = pd.read_csv(filename)
    x = file[['startupPer','backupPer','startupDef','backupDef']]
    y = file['winper']
    lm = linear_model.LinearRegression()
    lm.fit(x,y)
    w1 = lm.coef_[0]
    w2 = lm.coef_[1]
    w3 = lm.coef_[2]
    w4 = lm.coef_[3]
    intercept = lm.intercept_

    return w1,w2,w3,w4,intercept

if __name__ == '__main__':
    w1,w2,w3,w4,intercept = linearregression()
    lineUpPositions=['S','B']
    positionsCategory=['C','PF','PG','SF','SG']




    


