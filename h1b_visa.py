# data analysis
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# read the data
df = pd.read_csv('../input/h1b_kaggle.csv')

# head of the data
df.head()


# how to search
df.loc[df['EMPLOYER_NAME'] == 'UNIVERSITY OF MICHIGAN']
df.loc[df['column_name'].isin(some_values)]
df.loc[(df['column_name'] == some_value) & df['other_column'].isin(some_values)]

df.loc[df['EMPLOYER_NAME'].str.contains('UNIVERSITY', na=False)].head()
