import os
from sklearn.model_selection import train_test_split
from pre_processing import *
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

if os.path.exists("LRmodel.pkl"):
    testData = pd.read_csv("player-test-samples.csv")
    removenull = pd.notnull(testData)
    testData[removenull]
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    # actual y
    y = testData["PlayerLevel"]
    x, y = preProcessing(testData, x, y)

    model = pickle.load(open('LRmodel.pkl', 'rb'))  # load the trained model   #############
    model.predict(x)
    y_pred = model.predict(x)
    accuracy = np.mean(y_pred == y) * 100
    print("The achieved accuracy using Logistic Regression is " + str(accuracy))
    true_player_value = np.asarray(y)[0]
    predicted_player_value = y_pred[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))

else:
    data = pd.read_csv("player-classification.csv")
    pd.set_option('display.max_rows', None)
    X = data.iloc[:, :-1]
    # actual y
    Y = data["PlayerLevel"]
    X, Y = preProcessing(data, X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=0)

    # Create a Gaussian Classifier
    model = LogisticRegression(solver='liblinear', random_state=0, intercept_scaling=3)

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Predict Output
    y_prediction = model.predict(X_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Logistic Regression is " + str(accuracy))
    true_player_value = np.asarray(y_test)[0]
    predicted_player_value = y_prediction[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
    print(model.score(X_test, y_test) * 100)

    pickle.dump(model, open('LRmodel.pkl', 'wb'))  # save the trained model   #############



