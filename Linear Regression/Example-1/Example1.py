import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import pyplot
from matplotlib import style

# the label that needs to be predicted
predict = 'y'
# the attribute in the dataset
attribute = 'x'

# reading the csv data
data = pd.read_csv("dataSet.csv")

# preparing X ( set of attributes )
X = np.array(data.drop([predict], 1))

# preparing Y ( set of labels )
Y = np.array(data[predict])

# preparing test and train data
# test_size : the amount of data to be set aside for testing
# in this case around 10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# creating a linear regression model
model = linear_model.LinearRegression()
# training the model on the train data
model.fit(x_train, y_train)

# calculating the accuracy of the model
accuracy = model.score(x_test, y_test)
print("Accuracy: ", accuracy)

# displaying the computed intercept and co-efficients
print("Intercept: ", model.intercept_)
print("Co-Efficients: ", model.coef_)

# making predictions
predictions = model.predict(x_test)
# displaying predictions
for i in range(len(predictions)):
    print("Input:", x_test[i], " Expected: ", y_test[i], " Predicted: ", predictions[i])

# plotting to see the data points
style.use("ggplot")
pyplot.scatter(data[attribute], data[predict])
pyplot.xlabel("X")
pyplot.ylabel("Y")
pyplot.show()
