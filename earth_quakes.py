import pandas as pd
import numpy as np

import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

df = pd.read_csv('../input/database.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df['Type'].unique())
# print(df.columns)

# column_to_keep = ["Date", "Latitude", "Longitude", "Magnitude", "Depth", "Type"]
# df_keep = df[column_to_keep].sort_values(by = 'Magnitude', ascending = False)
# print(df_keep.info())
# df_keep["Date"] = pd.to_datetime(df_keep["Date"])
# print(df_keep.info())

# print(df_keep.pivot_table(index = "Type", values = "Magnitude", aggfunc=len))

# m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

# longitudes = df_keep["Longitude"].tolist()
# latitudes = df_keep["Latitude"].tolist()
# mags = df_keep["Magnitude"].tolist()
# x,y = m(longitudes,latitudes)

# fig = plt.figure(figsize=(12,10))
# plt.title("All Affected Areas")
# m.plot(x, y, "o", markersize = 3, color = 'blue')
# m.drawcoastlines()
# m.fillcontinents(color='coral',lake_color='aqua')
# m.drawmapboundary()
# m.drawcountries()
# plt.show()

## https://www.kaggle.com/artimous/d/usgs/earthquake-database/visualizing-earthquakes-via-animations
df['Year']= df['Date'].str[6:]

fig = plt.figure(figsize=(10, 10))
fig.text(.8, .3, 'Will', ha='right')
cmap = plt.get_cmap('coolwarm')

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='burlywood',lake_color='lightblue', zorder = 1)
m.drawmapboundary(fill_color='lightblue')


START_YEAR = 1965
LAST_YEAR = 2016

points = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][df['Year']==str(START_YEAR)]

x, y= m(list(points['Longitude']), list(points['Latitude']))
scat = m.scatter(x, y, s = points['Magnitude']/points['Depth']*10, marker='o', alpha=0.3, zorder=10, cmap = cmap)
year_text = plt.text(-170, 120, str(START_YEAR),fontsize=15)
plt.title("Earthquake visualisation (1965 - 2016)")
plt.close()

def update(frame_number):
    current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
    year_text.set_text(str(current_year))
    points = df[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']][t_file['Year']==str(current_year)]
    x, y= m(list(points['Longitude']), list(points['Latitude']))
    color = points['Depth']/points['Magnitude']*10;
    scat.set_offsets(np.dstack((x, y)))
    scat.set_sizes(points['Magnitude']/points['Depth']*10)

ani = animation.FuncAnimation(fig, update, interval=1000, frames=LAST_YEAR - START_YEAR + 1)
ani.save('animation.gif', writer='imagemagick', fps=5)

import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
