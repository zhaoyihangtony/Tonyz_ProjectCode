import pandas as pd    

seasons = ['2011','2012','2013','2014','2015','2016','2017']
for i in seasons:
    gamesfilename = 'game_'+ i + '.csv'
    gamesfile = pd.read_csv('FileGenerate/TimeSeries/' + gamesfilename)

    gamesfile.index = pd.to_datetime(gamesfile['date'])

    gamesfile.to_csv('FileGenerate/TimeSeries/'+ 'games_timefixed_'+i+'.csv')


    #print(gamesfile.index)
