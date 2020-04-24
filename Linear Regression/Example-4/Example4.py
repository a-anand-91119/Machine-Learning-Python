import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from matplotlib import pyplot
from matplotlib import style

# the data headers in file as as below
# No, X1 transaction date, X2 house age, X3 distance to the nearest MRT station,
# X4 number of convenience stores, X5 latitude, X6 longitude, Y house price of unit area
#
# we don't want out labels to be names like that, so we are gonna use pandas to rename the headers
# and also we are not going to read the column "No" and "X1 transaction date"
#
# How to skip a column while reading?
# First we are gonna read the headers only from the file
header = pd.read_csv("Real estate.csv", nrows=0)
print(header.columns)

# Now we are gonna use List Comprehension on this header and usecols attribute of pandas read_csv
data = pd.read_csv("Real estate.csv",
                   usecols=[column_name for column_name in header.columns
                            if (column_name != "No" and column_name != "X1 transaction date")])
print(data.head())

# now that the data has been read, lets replace the headers with more simpler ones
data.columns = ["house_age", "mrt_station", "conv_store", "latitude", "longitude", "price"]
print(data.head())

# label to be predicted
predict = "price"

# preparing attributes and labels array
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# create test and train datasets
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.1)

# create a model
model = linear_model.LinearRegression()

# train the model
model.fit(x_train, y_train)

# calculate the accuracy of the model
accuracy = model.score(x_test, y_test)
print("Model Accuracy: ", accuracy)

# print the model trained parameters
print("Intercept: ", model.intercept_)
print("Co-Efficients: ", model.coef_)

# make predictions
predictions = model.predict(x_test)

# limiting the number of elements displayed to console by np arrays
np.set_printoptions(threshold=2)

# displaying the predictions
print("Expected Output\tPredicted Output")
for i in range(len(predictions)):
    print(y_test[i], "\t", predictions[i])

style.use("ggplot")
attribute = "longitude"
pyplot.scatter(data[attribute], data[predict])
pyplot.xlabel(attribute)
pyplot.ylabel(predict)
pyplot.show()
