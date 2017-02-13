import pandas
import math
from sklearn.linear_model import LinearRegression
import numpy as np

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

sales = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_data.csv", dtype=dtype_dict)
test_data = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_test_data.csv", dtype=dtype_dict)
training_data = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_train_data.csv", dtype=dtype_dict)

training_data["bedrooms_squared"] = training_data["bedrooms"] ** 2
training_data["bed_bath_rooms"] = training_data["bedrooms"] * training_data["bathrooms"]
training_data["log_sqft_living"] = training_data["sqft_living"].apply(np.log)
training_data["lat_plus_long"]  = training_data["lat"] + training_data["long"]

test_data["bedrooms_squared"] = test_data["bedrooms"] ** 2
test_data["bed_bath_rooms"] = test_data["bedrooms"] * test_data["bathrooms"]
test_data["log_sqft_living"] = test_data["sqft_living"].apply(np.log)
test_data["lat_plus_long"]  = test_data["lat"] + test_data["long"]

# QUESTION 1
# print(test_data["bedrooms_squared"].mean())
# print(test_data["bed_bath_rooms"].mean())
# print(test_data["log_sqft_living"].mean())
# print(test_data["lat_plus_long"].mean())


train_data = pandas.DataFrame(training_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long"])
train_price = pandas.DataFrame(training_data, columns = ["price"])

model_1 = LinearRegression()
model_1.fit(train_data, train_price)
model_1_coef = pandas.DataFrame(model_1.coef_, columns = ["sqft_living", "bedrooms", "bathrooms", "lat", "long"])
print(model_1_coef)

train_data = pandas.DataFrame(training_data, columns= [""])
model_2 = LinearRegression()
model_1_coef.fit()

# df1 = pandas.DataFrame([1,1,1])
# df2 = pandas.DataFrame([1,1,1])
# df3 = pandas.DataFrame([1,1,1])
# df4 = pandas.concat([df1, df2, df3], axis=1,ignore_index= True)
# print(df4)
# print(df4.shape)
#
# df5 = pandas.DataFrame([1,1,1])
# # df5.resize((1,3))
# print(df5.values)
# print(df5.shape)
#
# regr = LinearRegression()
# regr.fit(df4, df5.values)
# print(regr.coef_)
# df6 = pandas.DataFrame([1,1,1])
# test = df6.values
# print(test.shape)
# test = test.resize((1,3))
#
# print(regr.predict(test))