import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Please find csv link at https://www.coursera.org/learn/ml-foundations/notebook/et8PR/predicting-house-prices-assignment

# importing data into python
data1=pd.read_csv("home_data.csv")

# Date column needs to be converted to datetime
print(type(data1['date']))
#print(pd.to_datetime(data1['date']).head())

data=pd.read_csv("home_data.csv",parse_dates=['date'])
print(type(data['date']))
#print(data.head())

# check the shape of the DataFrame (rows, columns)
print(data.shape)

#####  summary of the data

print (data.describe())


# Data split into test and train_test_split
train_dataPandas, test_dataPandas = train_test_split(data, train_size=0.8, random_state=1)
print(train_dataPandas.shape)
print(test_dataPandas.shape)


reg_model_Pandas = LinearRegression()
X = train_dataPandas[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]
y = train_dataPandas['price']

# Building model
reg_model_Pandas.fit(X, y)
