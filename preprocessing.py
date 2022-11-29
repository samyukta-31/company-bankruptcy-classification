# Importing dependencies
import pandas as pd


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