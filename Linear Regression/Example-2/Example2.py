import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import style
from matplotlib import pyplot

# limiting the number of elements displayed to console by np arrays
np.set_printoptions(threshold=3)

# prediction label
predict = "price"

# reading data from csv
data = pd.read_csv("CarPrice_Assignment.csv")

# trimming only the required attributes
# considering only attributes with numeric data
data = data[["wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm", "citympg", "highwaympg", "price"]]

# creating labels and attributes arrays
# attributes are unique to each car
# labels are what we derive from the attributes
# X is the set of attributes
X = np.array(data.drop([predict], 1))
# Y is the labels
Y = np.array(data[predict])

# creating test and train data using sklearn model selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# creating a linear regression model
# setting n_jobs as 2 to speed up the training process
model = linear_model.LinearRegression(n_jobs=2)

# training the model
model.fit(x_train, y_train)

# calculating the accuracy of the model
accuracy = model.score(x_test, y_test)
print("Model Accuracy: ", accuracy)

# getting the model intercept and co-efficients
print("Intercept: ", model.intercept_)
print("Co-Efficients: ", model.coef_)

# using model to make predictions
predictions = model.predict(x_test)

# displaying the predictions
for i in range(len(predictions)):
    print("Input: ", x_test[i], "Expected Output: ", y_test[i], "Predicted: ", predictions[i])

# plotting the attributes against the price
attribute = "curbweight"
style.use("ggplot")
pyplot.scatter(data[attribute], data[predict])
pyplot.ylabel(predict)
pyplot.xlabel(attribute)

pyplot.show()
