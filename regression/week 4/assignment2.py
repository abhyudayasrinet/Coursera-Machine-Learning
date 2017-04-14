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


def get_numpy_data(data_frame, features, output):
    data_frame['constant'] = 1
    features = ['constant'] + features
    features_matrix = np.matrix(data_frame.as_matrix(columns=features))
    output_array = data_frame.as_matrix(columns = [output])
    return features_matrix, output_array


def predict_output(feature_matrix, weights):
    # print("predict_output")
    # print(feature_matrix.shape)
    # print(weights.shape)
    predictions = np.dot(feature_matrix, weights)
    # print(predictions.shape)
    return predictions


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    errors = errors.transpose()
    if(feature_is_constant):
        derivative = 2*np.dot(errors, feature)
    else:
        derivative = 2*l2_penalty*weight + 2*np.dot(errors, feature)
    return derivative


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
        #while not reached maximum number of iterations:
        # compute the predictions using your predict_output() function
    # max_iterations=1
    for itr in range(max_iterations):
        predictions = predict_output(feature_matrix, weights)
        predictions = predictions.transpose()
        # print("predictions", predictions.shape)
        # print("output", output.shape)
        errors = np.subtract(predictions, output)
        # print("errors", errors.shape)
        # compute the errors as predictions - output
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            if(i==0):
                derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, False)

            weights[i] = weights[i] - step_size*derivative
            # subtract the step size times the derivative from the current weight
    return weights



(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
test_predictions = test_predictions.transpose()
errors = np.subtract(test_predictions, example_output) # prediction errors

# # next two lines should print the same values
# print(feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False))
# print(np.sum(errors.transpose()*example_features[:, 1])*2+20)
# print('')
#
# # next two lines should print the same values
# print(feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True))
# print(np.sum(errors)*2)

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(training_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
step_size = 1e-12
max_iterations = 1000
initial_weights = np.array([0., 0.])

l2_penalty = 0
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
#QUIZ QUESTION
print(simple_weights_0_penalty)

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
#QUIZ QUESTION
print(simple_weights_high_penalty)

#QUIZ QUESTION
# plt.plot(simple_feature_matrix,output,'k.',
#         simple_feature_matrix[:,1],predict_output(simple_feature_matrix, simple_weights_0_penalty).transpose(),'b-',
#         simple_feature_matrix[:,1],predict_output(simple_feature_matrix, simple_weights_high_penalty).transpose(),'r-')
# plt.show()


#QUIZ QUESTION
predictions = predict_output(simple_test_feature_matrix, initial_weights)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

predictions = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

predictions = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(training_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
initial_weights = np.array([0., 0., 0.])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
#QUIZ QUESTION
print(multiple_weights_0_penalty)

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations)
#QUIZ QUESTION
print(multiple_weights_high_penalty)

#QUIZ QUESTION
predictions = predict_output(test_feature_matrix, initial_weights)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

predictions = predict_output(test_feature_matrix, multiple_weights_0_penalty)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

predictions = predict_output(test_feature_matrix, multiple_weights_high_penalty)
predictions = predictions.transpose()
RSS = np.square(np.subtract(predictions, test_output)).sum()
print(RSS)

#QUIZ QUESTION
prediction = predict_output(test_feature_matrix[0,:], multiple_weights_0_penalty)
print(prediction)
prediction = predict_output(test_feature_matrix[0,:], multiple_weights_high_penalty)
print(prediction)