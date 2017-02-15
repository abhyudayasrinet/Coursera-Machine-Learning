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

# QUIZ QUESTION
# print(test_data["bedrooms_squared"].mean())
# print(test_data["bed_bath_rooms"].mean())
# print(test_data["log_sqft_living"].mean())
# print(test_data["lat_plus_long"].mean())

train_price = pandas.DataFrame(training_data, columns = ["price"])
test_price = pandas.DataFrame(test_data, columns= ["price"])

train_data1 = pandas.DataFrame(training_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long"])
model_1 = LinearRegression()
model_1.fit(train_data1, train_price)
model_1_coef = pandas.DataFrame(model_1.coef_, columns = ["sqft_living", "bedrooms", "bathrooms", "lat", "long"])
# QUIZ QUESTION
# print(model_1_coef)

train_data2 = pandas.DataFrame(training_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms"])
model_2 = LinearRegression()
model_2.fit(train_data2, train_price)
model_2_coef = pandas.DataFrame(model_2.coef_, columns = ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms"])
# QUIZ QUESTION
# print(model_2_coef)

train_data3 = pandas.DataFrame(training_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms", "bedrooms_squared", "log_sqft_living", "lat_plus_long"])
model_3 = LinearRegression()
model_3.fit(train_data3, train_price)
model_3_coef = pandas.DataFrame(model_3.coef_, columns = ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms", "bedrooms_squared", "log_sqft_living", "lat_plus_long"])
# QUIZ QUESTION
# print(model_3_coef)

test_data1 = pandas.DataFrame(test_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long"])
predictions_1 = model_1.predict(test_data1)
RSS_1 = ((predictions_1 - test_price)**2).sum()
# QUIZ QUESTION
# print(RSS_1)

test_data2 = pandas.DataFrame(test_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms"])
predictions_2 = model_2.predict(test_data2)
RSS_2 = ((predictions_2 - test_price)**2).sum()
# QUIZ QUESTION
# print(RSS_2)

test_data3 = pandas.DataFrame(test_data, columns= ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms", "bedrooms_squared", "log_sqft_living", "lat_plus_long"])
predictions_3 = model_3.predict(test_data3)
RSS_3 = ((predictions_3 - test_price)**2).sum()
# QUIZ QUESTION
# print(RSS_3)

predictions_1_train = model_1.predict(train_data1)
RSS_1_train = ((train_price - predictions_1_train)**2).sum()
# QUIZ QUESTION
# print(RSS_1_train)

predictions_2_train = model_2.predict(train_data2)
RSS_2_train = ((train_price - predictions_2_train)**2).sum()
# QUIZ QUESTION
# print(RSS_2_train)

predictions_3_train = model_3.predict(train_data3)
RSS_3_train = ((train_price - predictions_3_train)**2).sum()
# QUIZ QUESTION
# print(RSS_3_train)
