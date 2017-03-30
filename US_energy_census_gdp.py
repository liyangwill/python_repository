import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(r'../input/Energy Census and Economic Data US 2010-2014.csv')
print (df.head())
print (df.info())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

plt.scatter(df['GDP2014'], df['TotalPrice2014'], s=np.power(df['CENSUS2010POP'],0.3), alpha=0.5)
plt.xlabel('2014 GDP')
plt.ylabel('2014 Total Price')
plt.show()

temp = df[['StateCodes', 'GDP2014', 'TotalPrice2014', 'CENSUS2010POP']]
outlier1 = temp[(temp.GDP2014>15000000)]
outlier2 = temp[(temp.TotalPrice2014>35)]
print (outlier1)
print (outlier2)

temp = temp[(temp.GDP2014<15000000) & (temp.TotalPrice2014<35) & (temp.StateCodes != 'US') ]
plt.scatter(temp['GDP2014'], temp['TotalPrice2014'], s=np.power( temp['CENSUS2010POP'], 0.35), alpha=0.5)
plt.show()