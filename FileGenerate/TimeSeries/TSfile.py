import pandas as pd 


seasons = ['2011','2012','2013','2014','2015','2016','2017']
for i in seasons:
    gamesfilename = 'games'+ i + '.csv'
    gamesfile = pd.read_csv('FileGenerate/' + gamesfilename,sep='\t')

    gamesfile['WINrate'] = gamesfile['home_team_score']/gamesfile['visit_team_score']
    print(gamesfile)

    gamesfile.to_csv('FileGenerate/TimeSeries/'+'game_'+i+'.csv')


    
