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

sales = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_data.csv", dtype=dtype_dict)
test_data = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_test_data.csv", dtype=dtype_dict)
training_data = pandas.read_csv("~/PycharmProjects/ml/regression/kc_house_train_data.csv", dtype=dtype_dict)


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


def feature_derivative(errors, feature):
    derivative = 2 * np.multiply(errors, feature).sum()
    return derivative


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    # x_axis = []
    # y_axis = []
    # count = 0
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        # print("errors : " , errors.sum())
        # print("weights : ", weights)
        # # y_axis.append(weights[1,0])
        # x_axis.append(count)
        # count+=1
        gradient_sum_squares = 0  # initialize the gradient

        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            # print(errors)
            # print(feature_matrix[:,i])
            derivative = feature_derivative(errors, feature_matrix[:,i])
            # print(i,derivative)
            # input()
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += (derivative**2)
            # update the weight based on step size and derivative:
            weights[i] = weights[i] - step_size*derivative


        gradient_magnitude = np.sqrt(gradient_sum_squares)
        # y_axis.append(gradient_magnitude)
        # print("grad_mag : ",gradient_magnitude)
        # x = input()
        # if(x!=''):
        #     plt.plot(x_axis,y_axis,'ro')
        #     plt.show()
        if gradient_magnitude < tolerance:
            converged = True
    return weights

# TEST PREDICT OUTPUT FUNCTION
# (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
# my_weights = np.array([1., 1.])
# test_predictions = predict_output(example_features, my_weights)
# print(test_predictions)

#TEST DERIVATIVE FUNCTION
# (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
# my_weights = np.array([0., 0.]) # this makes all the predictions 0
# my_weights = my_weights.reshape((-1,1))
# test_predictions = predict_output(example_features, my_weights)
# print(test_predictions)
# # just like SFrames 2 numpy arrays can be elementwise subtracted with '-':
# errors = test_predictions - example_output # prediction errors in this case is just the -example_output
# print(errors)
# feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
# derivative = feature_derivative(errors, feature)
# print(derivative)
# print(-np.sum(example_output)*2) # should be the same as derivative


simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(training_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
initial_weights = initial_weights.reshape((-1,1))
step_size = 7e-12
tolerance = 2.5e7
simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
#QUIZ QUESTION
print("weight of sqft_living : " , simple_weights[1,0])
print(simple_weights)

test_simple_feature_matrix, output = get_numpy_data(test_data, simple_features, my_output)
predictions = predict_output(test_simple_feature_matrix, simple_weights)
#QUIZ QUESTION
print("model 1 predicted price of 1st house ",predictions[0])
print(predictions)

errors = predictions - output
RSS = (np.multiply(errors, errors)).sum()
#QUIZ QUESTION
print("RSS of model 1 : ", RSS)
print(RSS)


model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(training_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
initial_weights = initial_weights.reshape((-1,1))
step_size = 4e-12
tolerance = 1e9
new_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(new_weights)


test_simple_feature_matrix, output = get_numpy_data(test_data, model_features, my_output)
new_predictions = predict_output(test_simple_feature_matrix, new_weights)
#QUIZ QUESTION
print("model 2 predicted price of house 1 : " , new_predictions[0])
print(output[0])

errors = new_predictions - output
RSS_2 = np.multiply(errors, errors).sum()
print("RSS of model 2 : " , RSS_2)