import pandas as pd
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import matplotlib.pyplot as plt

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv(r"kc_house_data.csv", dtype=dtype_dict)
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)


def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features

    # features_frame = pandas.DataFrame(data_frame, columns=features)
    # features_matrix = features_frame.as_matrix()
    features_matrix = np.matrix(data_frame.as_matrix(columns=features))

    # output_frame = pandas.DataFrame(data_frame, columns=[output])
    # output_array = output_frame.as_matrix()
    output_array = data_frame.as_matrix(columns = [output])

    return features_matrix, output_array

def predict_output(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return predictions


def normalize_features(features):

    # print("features")
    # print(features)
    norms = np.linalg.norm(features, axis=0)
    # print(norms)
    normalized_features= features / norms
    return (normalized_features, norms)


simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(training, simple_features, my_output)
normalized_features, norms = normalize_features(simple_feature_matrix)
print(normalized_features)