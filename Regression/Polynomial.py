import os
import joblib
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from preprocessing import *
from sklearn.metrics import r2_score
import time
import warnings

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

if os.path.exists("polyModel.pkl") and os.path.exists("'poly_features_model'.pkl"):
    testData = pd.read_csv("player-test-samples.csv")
    removenull = pd.notnull(testData)
    testData[removenull]
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    # actual y
    y = testData["value"]
    x, y = preProcessing(testData, x, y)

    # load the trained model
    poly_features_model = joblib.load('poly_features_model')
    poly_model = joblib.load('poly_model')

    # predicting on test data-set
    X_poly_prep = poly_features_model.transform(x)
    predictions = poly_model.predict(X_poly_prep)
    print('Mean Square Error With Polynomial Model', metrics.mean_squared_error(y, predictions))

    true_player_value = np.asarray(y)[0]
    predicted_player_value = predictions[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value/10**6))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value/10**6))
    print("Accuracy:", r2_score(y, predictions))

else:
    # Bilding Model

    data = pd.read_csv("player-value-prediction.csv")
    pd.set_option('display.max_rows', None)
    X = data.iloc[:, :-1]
    # actual y
    Y = data["value"]

    X, Y = preProcessing(data, X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)
    poly_features = PolynomialFeatures(degree=2)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    #X_train_poly = poly_model.fit_transform(X_train)

    # Training Time
    start = time.time()
    poly_model.fit(X_train_poly, y_train)
    stop = time.time()
    print("training time = ", stop - start, " sec.")

    # save the trained model
    joblib.dump(poly_model, 'poly_model')
    joblib.dump(poly_features, 'poly_features_model')

    y_train_predicted = poly_model.predict(X_train_poly)
    ypred = poly_model.predict(poly_features.transform(X_test))

    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    print('Mean Square Error With Polynomial Model', metrics.mean_squared_error(y_test, prediction))

    true_player_value = np.asarray(y_test)[0]
    predicted_player_value = prediction[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value/pow(10,6)))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value/10**6))
    print("Accuracy:", r2_score(y_test, prediction))


