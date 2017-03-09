import pandas
import math
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


dtype_dict = {'bathrooms':float,
              'waterfront':int,
              'sqft_above':int,
              'sqft_living15':float,
              'grade':int,
              'yr_renovated':int,
              'price':float,
              'bedrooms':float,
              'zipcode':str,
              'long':float,
              'sqft_lot15':float,
              'sqft_living':float,
              'floors':str,
              'condition':int,
              'lat':float,
              'date':str,
              'sqft_basement':int,
              'yr_built':int,
              'id':str,
              'sqft_lot':int,
              'view':int}

sales = pandas.read_csv("C:\\Users\\absrin\\PycharmProjects\\ml\\regression\\week 3\\kc_house_data.csv", dtype=dtype_dict)
sales = sales.sort_values(by=['sqft_living','price'])
test_data = pandas.read_csv("C:\\Users\\absrin\\PycharmProjects\\ml\\regression\\week 3\\wk3_kc_house_test_data.csv", dtype=dtype_dict)
training_data = pandas.read_csv("C:\\Users\\absrin\\PycharmProjects\\ml\\regression\\week 3\\wk3_kc_house_train_data.csv", dtype=dtype_dict)
validation_data = pandas.read_csv("C:\\Users\\absrin\\PycharmProjects\\ml\\regression\\week 3\\wk3_kc_house_valid_data.csv", dtype=dtype_dict)


def polynomial_dataframe(feature, degree):
    poly_dataframe = pandas.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature**power
    return poly_dataframe


poly1_data = polynomial_dataframe(sales["sqft_living"], 1)
poly1_data["price"] = sales["price"]
model1 = LinearRegression()
model1.fit(poly1_data, sales["price"])
print(model1.coef_)


plt.plot(poly1_data['power_1'],poly1_data['price'],'.',poly1_data['power_1'], model1.predict(poly1_data),'-')
plt.show()