import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from matplotlib import pyplot
from matplotlib import style

# prediction variable
predict = "Salary"

# because there's only one attribute, using a variable for it
attribute = "YearsExperience"

# reading data
data = pd.read_csv("Salary_Data.csv")

# creating labels and attributes array
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# creating test and train data sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# creating a model
model = linear_model.LinearRegression()

# training the model
model.fit(x_train, y_train)

# calculating the accuracy of the model
accuracy = model.score(x_test, y_test)
print("Model Accuracy: ", accuracy)

# displaying the model parameters
print("Intercept: ", model.intercept_)
print("Co-Efficients: ", model.coef_)

# making predictions
predictions = model.predict(x_test)

# displaying the predictions
print("Input\tExpected Output\tPredicted Output")
for i in range(len(predictions)):
    print(x_test[i], "\t", y_test[i], '\t', predictions[i])

# plotting the attribute and labels
style.use("ggplot")
pyplot.scatter(data[attribute], data[predict])
pyplot.xlabel("Years Of Experience")
pyplot.ylabel("Salary")
pyplot.show()
