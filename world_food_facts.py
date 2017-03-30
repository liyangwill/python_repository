import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/en.openfoodfacts.org.products.tsv", sep='\t',low_memory=False)

# print(df.head())
# print(df.info())
# print(df.describe())

df.countries = df.countries.str.lower()

def mean(l):
    return float(sum(l)) / len(l)

df_sugars = df[df.sugars_100g.notnull()]

def return_sugars(country):
    return df_sugars[df_sugars.countries == country].sugars_100g.tolist()

# Get list of sugars per 100g for some countries
fr_sugars = return_sugars('france') + return_sugars('en:fr')
za_sugars = return_sugars('south africa')
uk_sugars = return_sugars('united kingdom') + return_sugars('en:gb')
us_sugars = return_sugars('united states') + return_sugars('en:us') + return_sugars('us')
sp_sugars = return_sugars('spain') + return_sugars('en:es')
nd_sugars = return_sugars('netherlands') + return_sugars('holland')
au_sugars = return_sugars('australia') + return_sugars('en:au')
cn_sugars = return_sugars('canada') + return_sugars('en:cn')
de_sugars = return_sugars('germany')

countries = ['FR', 'ZA', 'UK', 'US', 'ES', 'ND', 'AU', 'CN', 'DE']
sugars_l = [mean(fr_sugars),
            mean(za_sugars),
            mean(uk_sugars),
            mean(us_sugars),
            mean(sp_sugars),
            mean(nd_sugars),
            mean(au_sugars),
            mean(cn_sugars),
            mean(de_sugars)]

x_pos = np.arange(len(countries))

plt.bar(y_pos, sugars_l, align='center', alpha=0.5)
plt.title('Average total sugar content per 100g')
plt.xticks(x_pos, countries)
plt.ylabel('Sugar/100g')

plt.show()