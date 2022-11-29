# Importing dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

### Pre-processing and Exploratpry Data Analysis ### 

# Reading data from csv and saving to a dataframe
raw_data = pd.read_csv('./data/data.csv')

# Data dimensions
raw_data.shape

# Data information
raw_data.info()

# Top 5 data entries
raw_data.head()

# Checking for missing data
raw_data.isnull().sum()

# Checking for data imbalance
raw_data["Bankrupt?"].value_counts()

# Checking for duplicate data 
raw_data.duplicated().sum()

# Drop net income flag as all entries are 1, giving no useful information for model training
raw_data[' Net Income Flag'].value_counts()

raw_data = raw_data.drop(' Net Income Flag',axis=1)

# Separating features and label
label = raw_data.columns[0]

features = raw_data.columns[1:]

X, Y = raw_data[features], raw_data[label]

# Split data 80-20 into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))

train_data = pd.concat((y_train,X_train),axis=1)

test_data = pd.concat((y_test, X_test),axis=1)

# Writing the training and testing data into seperate csv files
train_data.to_csv('./data/train.csv', index=False)

train_data.to_csv('./data/test.csv', index=False)