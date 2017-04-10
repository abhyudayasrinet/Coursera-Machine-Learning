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
test_data = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\kc_house_test_data.csv", dtype=dtype_dict)
training_data = pd.read_csv(r"C:\Users\abhyu\PycharmProjects\Coursera-Machine-Learning\regression\week 4\kc_house_train_data.csv", dtype=dtype_dict)
