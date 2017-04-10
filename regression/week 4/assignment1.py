import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge
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

sales = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 3\kc_house_data.csv", dtype=dtype_dict)

sales = sales.sort_values(by=['sqft_living','price'])
sales_price = sales["price"]
test_data = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\wk3_kc_house_test_data.csv", dtype=dtype_dict)
training_data = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\wk3_kc_house_train_data.csv", dtype=dtype_dict)
validation_data = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\wk3_kc_house_valid_data.csv", dtype=dtype_dict)



def polynomial_dataframe(feature, degree):
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature**power
    return poly_dataframe


# l2_small_penalty = 1.5e-5
#
# poly15_data = polynomial_dataframe(sales['sqft_living'], 15)
# model = Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, sales['price'])
# # QUESTION 1 - coefficient of power_1
# # print(model.coef_)
#
#
# l2_small_penalty = 1e-9
# set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
# set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
# set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
# set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)

# poly15_data = polynomial_dataframe(set_1["sqft_living"],15)
# poly15_data["intercept"] = pd.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model_1 = LinearRegression()
# model_1.fit(poly15_data, set_1["price"])
# plt.plot(poly15_data["power_1"], set_1["price"], '.', poly15_data["power_1"], model_1.predict(poly15_data), '-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_2["sqft_living"],15)
# poly15_data["intercept"] = pd.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model_1 = LinearRegression()
# model_1.fit(poly15_data, set_2["price"])
# plt.plot(poly15_data["power_1"], set_2["price"], '.', poly15_data["power_1"], model_1.predict(poly15_data), '-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_3["sqft_living"],15)
# poly15_data["intercept"] = pd.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model_1 = LinearRegression()
# model_1.fit(poly15_data, set_3["price"])
# plt.plot(poly15_data["power_1"], set_3["price"], '.', poly15_data["power_1"], model_1.predict(poly15_data), '-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_4["sqft_living"],15)
# poly15_data["intercept"] = pd.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model_1 = LinearRegression()
# model_1.fit(poly15_data, set_4["price"])
# plt.plot(poly15_data["power_1"], set_4["price"], '.', poly15_data["power_1"], model_1.predict(poly15_data), '-')
# plt.show()


# l2_small_penalty=1e1
#
# poly15_data = polynomial_dataframe(set_1['sqft_living'], 15)
# poly15_data["intercept"] = pd.DataFrame([1]*len(poly15_data), columns=["intercept"])
# model = Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, set_1['price'])
# # QUESTION 1 - coefficient of power_1
# print(model.coef_)
# plt.plot(poly15_data["power_1"], set_1["price"],'.',poly15_data["power_1"], model.predict((poly15_data)),'-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_2['sqft_living'], 15)
# model = Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, set_2['price'])
# # QUESTION 1 - coefficient of power_1
# print(model.coef_)
# plt.plot(poly15_data["power_1"], set_2["price"],'.',poly15_data["power_1"], model.predict((poly15_data)),'-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_3['sqft_living'], 15)
# model = Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, set_3['price'])
# # QUESTION 1 - coefficient of power_1
# print(model.coef_)
# plt.plot(poly15_data["power_1"], set_3["price"],'.',poly15_data["power_1"], model.predict((poly15_data)),'-')
# plt.show()
#
# poly15_data = polynomial_dataframe(set_4['sqft_living'], 15)
# model = Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, set_4['price'])
# # QUESTION 1 - coefficient of power_1
# print(model.coef_)
# plt.plot(poly15_data["power_1"], set_4["price"],'.',poly15_data["power_1"], model.predict((poly15_data)),'-')
# plt.show()

# train_valid_shuffled = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\wk3_kc_house_train_valid_shuffled.csv", dtype=dtype_dict)
# def k_fold_cross_validation(k, l2_penalty, data, output):
#     sum = 0
#     n = len(data)
#     for i in range(0,k):
#         start = int((n*i)/k)
#         end = int((n*(i+1))/k-1)
#         # print(start, end)
#         validation_set = data[start:end+1]
#         validation_output = output[start:end+1]
#         train_set = data[0:start].append(data[end:n])
#         output_set = output[0:start].append(output[end:n])
#         # print(train_set)
#         # print("====")
#         # print(output_set)
#         model = Ridge(alpha=l2_penalty, normalize=True)
#         model.fit(train_set, output_set)
#         predictions = model.predict(validation_set)
#         RSS = ((validation_output - predictions)**2).sum()
#         sum += RSS
#     return sum/float(n)
#
# data = polynomial_dataframe(train_valid_shuffled["sqft_living"], 15)
# # print(data[0:10])
# rss_list = []
# for i in  np.logspace(3, 9, num=13):
#     avg_rss = k_fold_cross_validation(10, i, data, train_valid_shuffled["price"])
#     rss_list.append(avg_rss)
#     print(i, avg_rss)
#
# #QUIZ QUESTION
# print(rss_list)
# print(min(rss_list))
#
# print(test_data)

l2_penalty = 1000
train_data = polynomial_dataframe(training_data["sqft_living"],15)
train_data["intercept"] = pd.DataFrame([1]*len(train_data), columns=["intercept"])
model = Ridge(alpha=l2_penalty, normalize=True)
model.fit(train_data, training_data["price"])
test_set = polynomial_dataframe(test_data["sqft_living"], 15)
test_set["intercept"] = pd.DataFrame([1]*len(test_set), columns=["intercept"])
test_output = test_data["price"]
predictions = model.predict(test_set)
RSS = ((test_output - predictions)**2).sum()
print(RSS)