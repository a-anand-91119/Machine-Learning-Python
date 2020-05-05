import pandas
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import pickle
import os.path

# reading the data from a csv file
data = pandas.read_csv("Salary_Data.csv")

# the label to predict
predict = "Salary"

# creating the attributes and labels array
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# before creating the model, check whether a saved model exists or not
# if it does, the load the model from the file and proceed to prediction
# otherwise create the model, train the model and save the model
if os.path.exists("trained_model.pickle"):
    print("Loading Trained Model")
    model = pickle.load(open("trained_model.pickle", "rb"))
else:
    print("Creating and training the model")
    print("**********************************")
    best_accuracy = 0
    # creating different versions of them model and saving only if new model has a better accuracy that previous ones
    for _ in range(100):
        # each time the model will be trained using a different test and train dataset
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

        # creating the model
        model = linear_model.LinearRegression()

        # training the model
        model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)
        if accuracy > best_accuracy:
            print("New Accuracy:", accuracy)
            best_accuracy = accuracy

            # once the model has been trained, we can save it to a file using pickle
            # open the target and and dump the model using pickle
            with open("trained_model.pickle", "wb") as file:
                pickle.dump(model, file)

    print("**********************************")
    print("Model Trained Successfully With An Accuracy:", best_accuracy)
    # need to load the best model from the file since newer ones might be less accurate
    model = pickle.load(open("trained_model.pickle", "rb"))

# calculating the accuracy of the model
accuracy = model.score(X, Y)
print("Model Accuracy: ", accuracy)

# displaying the model intercept and co-efficients
print("Intercept: ", model.intercept_)
print("Co-efficients:", model.coef_)

# making predictions using the model
predictions = model.predict(X)
# displaying the predictions
print("\nExperience\tActual Salary\tPredicted Salary")
for i in range(len(predictions)):
    print(X[i], "\t\t", Y[i], "\t\t", predictions[i])

# predicting a single value
experience = 25
new_prediction = model.predict([[experience]])
print("Experience: ", experience, " Predicted Salary: ", new_prediction[0])

