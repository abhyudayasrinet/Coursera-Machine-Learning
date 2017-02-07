import pandas


def simple_linear_regression(input_feature, output):
    pass
    sum_y = output.sum()
    sum_x = input_feature.sum()
    sum_xy = (input_feature * output).sum()
    sum_x2 = (input_feature ** 2).sum()
    N = output.size
    slope = ( (sum_xy - ((sum_x*sum_y)/N)) / (sum_x2 - ((sum_x*sum_x)/N)))
    intercept = sum_y/N - (slope * (sum_x / N))
    return intercept, slope


def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + slope*input_feature
    return predicted_output


def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    predictions = get_regression_predictions(input_feature, intercept, slope)
    RSS = ((output - predictions)**2).sum()
    return RSS


def inverse_regression_predictions(output, intercept, slope):
    estimated_input = ((output - intercept) / slope)
    return estimated_input

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

sales = pandas.read_csv("kc_house_data.csv", dtype = dtype_dict)
test_data = pandas.read_csv("kc_house_test_data.csv", dtype = dtype_dict)
train_data = pandas.read_csv("kc_house_train_data.csv", dtype = dtype_dict)

intercept, slope = simple_linear_regression(train_data["sqft_living"], train_data["price"])
sqft_intercept = intercept
sqft_slope = slope

print(intercept, slope)
print(get_regression_predictions(2650, intercept, slope))
sqft_RSS = get_residual_sum_of_squares(train_data["sqft_living"], train_data["price"], intercept, slope)
print(sqft_RSS)
print(inverse_regression_predictions(800000, intercept, slope))

bedroom_intercept, bedroom_slope = simple_linear_regression(train_data["bedrooms"], train_data["price"])
bedroom_RSS = get_residual_sum_of_squares(train_data["bedrooms"], train_data["price"], intercept, slope)
print(bedroom_RSS)

print("sqft" if bedroom_RSS > sqft_RSS else "bedrooms")