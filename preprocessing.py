# Importing dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

## Removing class imbalance with SMOTE
smote = SMOTE(random_state=50)

# Fitting predictor and target variable by applying the transform over unscaled train and test data
X_train, y_train = smote.fit_resample(X_train, y_train)
X_test, y_test = smote.fit_resample(X_test, y_test)

# Performing feature standardisation
sc = StandardScaler() # subtracts mean and divides by standard deviation to produce unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Applying PCA function on training and testing set of features
pca = PCA(n_components=0.80) # Setting expected variace instead of manually entering the no. of components to select
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
pca.explained_variance_ratio_ # Sum of variance of all components = 0.80 (advised between 0.8 and 0.85 to avoid overfitting)

# Feature dimensionality reduced from 94 to 32
X_train.shape , X_test.shape

# Writing the training and testing data into seperate csv files
train_data = pd.concat((pd.DataFrame(y_train),pd.DataFrame(X_train)),axis=1)
test_data = pd.concat((pd.DataFrame(y_test), pd.DataFrame(X_test)),axis=1)
train_data.to_csv('./data/train.csv', index=False)
test_data.to_csv('./data/test.csv', index=False)