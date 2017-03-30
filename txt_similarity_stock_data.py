import numpy as np
import pandas as pd

import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize, ngrams
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
# import xgboost as xgb

eng_stopwords = set(stopwords.words('english'))
color = sns.color_palette()
