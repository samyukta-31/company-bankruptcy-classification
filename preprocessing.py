# import dependeicies 
import pandas as pd
# from sklearn.model_selection import train_test_split

# read dataset and create dataframe
raw_data = pd.read_csv("./data/data.csv")

print(raw_data.head(10))