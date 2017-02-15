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


def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features

    features_frame = pandas.DataFrame(data_frame, columns=features)
    # features_matrix = features_frame.as_matrix()
    features_matrix = features_frame.values

    output_frame = pandas.DataFrame(data_frame, columns=[output])
    # output_array = output_frame.as_matrix()
    output_array = output_frame.values

    return features_matrix, output_array


def predict_outcome(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return predictions


def feature_derivative(errors, feature):
    derivative = 2 * np.dot(feature,errors)
    return derivative


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        predictions = predict_outcome(feature_matrix, weights)
        errors = output - predictions
        # print("errors")
        # print(errors)
        # print(errors.shape)
        # print("feature matrix")
        # print(feature_matrix)
        # print(feature_matrix.shape)
        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            # print(feature_matrix)
            matrix = feature_matrix[:,i]
            r = matrix.shape
            matrix = matrix.reshape(r[0],1)
            # print(errors[:].shape)
            # print(matrix.shape)

            derivative = feature_derivative(np.asarray(errors), np.asarray(feature_matrix[:,i]))
            # print(derivative)
            # break
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += (derivative**2)
            # update the weight based on step size and derivative:
            weights[i] = weights[i] + 2*step_size*derivative
        # break
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        print(weights)
        print(gradient_magnitude)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


# feature_matrix, output = get_numpy_data(training_data, ["sqft_living"],training_data["price"])
# # print(feature_matrix[:,1])
# # print(output)
#
#
# predictions = predict_outcome(np.array([1,2,3]), np.array([2,3,4]))
# print(predictions)

simple_features = ['sqft_living']
my_output= 'price'
simple_feature_matrix, output = get_numpy_data(training_data, simple_features, my_output)
initial_weights = np.array([1. ,-47000.])
initial_weights.resize((2,1))
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
print(simple_weights)