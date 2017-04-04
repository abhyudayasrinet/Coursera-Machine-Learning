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

sales = pandas.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 3\kc_house_data.csv", dtype=dtype_dict)

sales = sales.sort_values(by=['sqft_living','price'])
sales_price = sales["price"]
test_data = pandas.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 3\wk3_kc_house_test_data.csv", dtype=dtype_dict)
training_data = pandas.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 3\wk3_kc_house_train_data.csv", dtype=dtype_dict)
validation_data = pandas.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 3\wk3_kc_house_valid_data.csv", dtype=dtype_dict)


def polynomial_dataframe(feature, degree):
    poly_dataframe = pandas.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature**power
    return poly_dataframe

# create a dataframe
# df = pandas.DataFrame([1]*5, columns = ["intercept"])
# df2 = pandas.DataFrame([2]*5, columns = ["intercept"])
# print(df)
# print(df2)
# print(df-df2)

# POWER 1 POLYNOMIAL FIT
# poly1_data = polynomial_dataframe(sales["sqft_living"], 1)
# poly1_data["intercept"] = pandas.DataFrame([1]*len(poly1_data))
# model1 = LinearRegression()
# model1.fit(poly1_data, sales_price)
# print(model1.coef_)
# plt.plot(sales['sqft_living'],sales['price'],'.',poly1_data['power_1'], model1.predict(poly1_data),'-')
# # plt.plot(poly1_data['power_1'], model1.predict(poly1_data),'-')
# plt.show()


# POWER 2 POLYNOMAIAL FIT
# poly2_data = polynomial_dataframe(sales["sqft_living"], 2)
# poly2_data["intercept"] = pandas.DataFrame([1]*len(poly2_data), columns=["intercept"])
# model2 = LinearRegression()
# model2.fit(poly2_data, sales_price)
#
# plt.plot(sales["sqft_living"], sales["price"], '.', poly2_data["power_1"], model2.predict(poly2_data), '-')
# plt.show()


#POWER 3 POLYNOMIAL FIT
# poly3_data = polynomial_dataframe(sales["sqft_living"], 3)
# poly3_data["intercept"] = pandas.DataFrame([1] * len(poly3_data), columns=["intercept"])
# print(poly3_data)
# model3 = LinearRegression()
# model3.fit(poly3_data, sales_price)
# print(model3.coef_)
#
# plt.plot(sales["sqft_living"], sales["price"],'.', poly3_data["power_1"], model3.predict(poly3_data))
# plt.show()

# POWER 15 POLYNOMIAL FIT
# poly15_data = polynomial_dataframe(sales["sqft_living"], 15)
# poly15_data["intercept"] = pandas.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model15 = LinearRegression()
# model15.fit(poly15_data, sales_price)
# print(poly15_data)
# print(model15.coef_)
#
# plt.plot(sales["sqft_living"], sales["price"], ".", poly15_data["power_1"], model15.predict(poly15_data))
# plt.show()

for i in range(1,16):
    poly_data = polynomial_dataframe(training_data["sqft_living"],i)
    poly_data["intercept"] = pandas.DataFrame([1]*len(poly_data), columns=["intercept"])
    model = LinearRegression()
    model.fit(poly_data, training_data["price"])
    valid_data = polynomial_dataframe(validation_data["sqft_living"], i)
    valid_data["intercept"] = pandas.DataFrame([1]*len(valid_data), columns=["intercept"])
    # print(valid_data)
    predictions = model.predict(valid_data)
    RSS = ((predictions - validation_data["price"])**2).sum()
    print(i, RSS)

poly_data = polynomial_dataframe(training_data["sqft_living"], 6)
poly_data["intercept"] = pandas.DataFrame([1]*len(poly_data), columns=["intercept"])
model = LinearRegression()
model.fit(poly_data, training_data["price"])
print(model.coef_)
test_data_2 = polynomial_dataframe(test_data["sqft_living"], 6)
test_data_2["intercept"] = pandas.DataFrame([1]*len(test_data_2), columns=["intercept"])
predictions = model.predict(test_data_2)
RSS = ((predictions - test_data["price"])**2).sum()
print(RSS)

