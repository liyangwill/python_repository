##############################################################
######################### library ############################
##############################################################
# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


##############################################################
######################### EDA ################################
##############################################################
# read the data
df = pd.read_csv('file.csv')
dF = pd.read_csv(file_name, header=None)

# head of data
df.head()

# data sample
df.sample(10)

# num of entries, type of each column
df.info()

# statistical summary of numeric columns
df.describe()
# check categorical variables
df.describe(include = ['O'])

# size of the df
df.shape

# correlation between numeric volumns
df.corr()
## visualize correlation
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

# unique value in a column
df['COL1'].unique()

column_list = df.columns
# check null value status
df.isnull().sum()
# fillna data with column mean
df['COL1'] = df.COL1.fillna(df.COL1.mean())
# categorize column
df['COL1'] = df['COL1'].map({'VALUE1' : 1, 'VALUE2' : 0}).astype(int)


# group data and check statistics
print(df[['COL1', 'COL2']].groupby(['COL1'], as_index=False).mean().sort_values(by='COL2', ascending=False))
print(df.groupby('COL1').mean())
print(df.groupby('COL1').sum())

# Assign the new column labels to the DataFrame
columns_names=df.columns.tolist()
df.columns = column_labels_list

# check 'COL1', how many appearance of each value
df.['COL1'].value_counts()

# check missing values' quantity
df.isnull().sum()

# sort data by 'COL1'
df[['COL1', 'COL2']].sort_values(by = 'COL1', ascending = False).head(20)

# drop data
df.drop(['ROW1', 'ROW2'])
df.drop('COL1', axis=1)
df_dropped = df.drop(list_to_drop, axis='columns')
df = df[df.COL1 != 'VALUE1']

# Convert the COL1 to numeric values: df_clean['COL1']
df['COL1'] = pd.to_numeric(df['COL1'], errors='coerce')

# reset index
df = df.reset_index()['COL1']

# remove outliers
#### Removing the outliers
df = df[(df.COL1 <= VALUE1) & (df.COL2 >= VALUE2)]

# change date column format
df["Date"] = pd.to_datetime(df["Date"])

# pivot table summary
print(df.pivot_table(index = "COL1", values = "COL2", aggfunc=len))

##############################################################
######################### plots ##############################
##############################################################

# bar plots
df['COL1'].value_counts().plot(kind='bar')
df.COL1.value_counts().head(20).plot(kind = 'bar')
g = sns.countplot(df['COL1'])
# horizontal bar plot
plt.barh(y, CATEGORY, align='center', alpha=0.8)


# box plots
data.plot(kind='box')

# histogram
data.plot(kind='hist', bins=10, alpha=0.5, subplots=True)
# multiple histograms with sns
g = sns.FacetGrid(df, col='COL1')
g.map(plt.hist, 'COL2', bins=20)

# 2D scatter plot with marker 'o'
plt.plot(df.COL1, df.COL2, 'o', alpha = 0.2)

# two distributions in one plot
sns.distplot(df.COL1[df['COL1'] == 'VALUE1'], np.arange(LOW,HIGH,STEP))
sns.distplot(df.COL1[df['COL1'] == 'VALUE2'], np.arange(LOW,HIGH,STEP))

# axis, legend, and title
plt.xlim([LOW,HIGH])
plt.xlabel('X-AXIS LABEL')
plt.legend('VALUE1,'VALUE2')
plt.title('TITLE_TITLE_TITLE')
rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)


##############################################################
################## ML fit and predict ########################
##############################################################
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
