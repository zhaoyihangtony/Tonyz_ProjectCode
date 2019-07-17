import numpy as np
import pandas as pd
import requests
import csv
from bs4 import BeautifulSoup
from datetime import datetime, date

year = '2012'
filename = 'games' + year + '.csv'
games = pd.read_csv(filename, sep='\t')

BASE_URL = "http://www.espn.com/nba/game?gameId={0}"

match_id = []
dates = []
home_team = []
home_team_score = []
visit_team = []
visit_team_score = []
column1 = []
column2 = []
count = 0


for index, row in games.iterrows():
    geturl = BASE_URL.format(row['id'])
    r = requests.get(geturl)
    table = BeautifulSoup(r.text,'lxml').table
    match_id.append(row['id'])
   # print(table)
    count+= 1
    print(count)
    for i in table.find_all('tr')[1:]:        
       # print(i)
        columns = i.find_all('td')
       # print(columns)
       # print(columns[0].text)
        column1.append(columns[0].text)
        #print(columns[5].text)
        column2.append(columns[5].text)
    home_team.append(column1[1])
    visit_team.append(column1[0])
    home_team_score.append(column2[1])
    visit_team_score.append(column2[0])
    #print(match_id, home_team,home_team_score,visit_team,visit_team_score)
    del column1[:]
    del column2[:]

dic = {'id': match_id, 'home_team': home_team, 'visit_team': visit_team,
        'home_team_score': home_team_score, 'visit_team_score': visit_team_score}
file = pd.DataFrame(dic)
print(file)
file.to_csv('FileGenerate/games12.csv', sep='\t')

    
    #print(row)
    #columns = columns.find_all('td')
    #columns = np.array(columns)
    #columns.split('</td><td>')    



    
