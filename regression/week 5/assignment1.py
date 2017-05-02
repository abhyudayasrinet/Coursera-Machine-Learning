import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import matplotlib.pyplot as plt

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv(r"kc_house_data.csv", dtype=dtype_dict)

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = Lasso(alpha=5e2, normalize=True) # set parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
#QUIZ QUESTION
# print(model_all.coef_)


testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']


for l1_penalty in np.logspace(1, 7, num=13):
    model = Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training["price"])
    predictions = model.predict(training[all_features])
    RSS = ((predictions - training["price"])**2).sum()
    #QUIZ QUESTION
    # print(l1_penalty, RSS)

l1_penalty = 10
model = Lasso(alpha=l1_penalty, normalize=True)
model.fit(training[all_features], training["price"])
predictions = model.predict(testing[all_features])
RSS = ((predictions - testing["price"])**2).sum()
# print(model.coef_)
# print(model.intercept_)
#QUIZ QUESTION
# print(RSS)
# print(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))

l1_penalty_max = 0
l1_penalty_min = 0
max_nonzeros = 7
for l1_penalty in np.logspace(1,4, num=20):
    model = Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training["price"])
    nonzeros = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    if(nonzeros > max_nonzeros):
        l1_penalty_min = l1_penalty
    if(nonzeros < max_nonzeros):
        l1_penalty_max = l1_penalty
    # print(l1_penalty, nonzeros)

#QUIZ QUESTION
# print(l1_penalty_min)
# print(l1_penalty_max)


for l1_penalty in np.linspace(l1_penalty_min,l1_penalty_max,20):
    model = Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training["price"])
    predictions = model.predict(validation[all_features])
    RSS = ((predictions - validation["price"])**2).sum()
    nonzeros = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    print(l1_penalty, nonzeros, RSS)

l1_penalty = 127.42749857
model = Lasso(alpha=l1_penalty, normalize=True)
model.fit(training[all_features], training["price"])
print(model.coef_)