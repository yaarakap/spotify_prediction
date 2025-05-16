from  get_playlist_data import *
# from IPython.core.display import clear_output

import pandas as pd 
# import pandasql as ps 
# import time
# import sqlite3

tracks,columns = download_playlist('/0kUU2WZEpJkAK5r3kpDk69?si=49a5514365ba4105',50)
#If the id if for artist, you must to put specify True to the artist parameter
# tracks,columns = download_albums('id_of_the_artist_or_the_album',artist=False)
df1 = pd.DataFrame(tracks,columns=columns)
df1.head()

df1.to_csv('playlist1.csv',index=False)