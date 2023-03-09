import os
import time
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from preprocessing import *
from sklearn.metrics import r2_score
import xgboost
import pickle
import warnings

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

if os.path.exists("XGBmodel.pkl"):
    testData = pd.read_csv("player-test-samples.csv")
    removenull = pd.notnull(testData)
    testData[removenull]
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    # actual y
    y = testData["value"]
    x, y = preProcessing(testData, x, y)

    model = pickle.load(open('XGBmodel.pkl', 'rb'))  # load the trained model   #############
    model.predict(x)
    y_pred = model.predict(x)
    print('Mean Square Error with XGBOOST', metrics.mean_squared_error(y, y_pred))
    true_player_value = np.asarray(y)[0]
    predicted_player_value = y_pred[0]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
    print("Accuracy:", r2_score(y, y_pred))

else:
    # building model

    data = pd.read_csv("player-value-prediction.csv")
    pd.set_option('display.max_rows', None)
    X = data.iloc[:, :-1]
    # actual y
    Y = data["value"]
    X, Y = preProcessing(data, X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)
    model = xgboost.XGBRegressor(n_estimators=150, learning_rate=0.2, max_depth=10)

    # Training Time
    start = time.time()
    model.fit(x_train, y_train)
    stop = time.time()
    print("training time = ", stop - start, " sec.")
    y_pred = model.predict(x_test)
    print('Mean Square Error with XGBOOST', metrics.mean_squared_error(y_test, y_pred))
    true_player_value = np.asarray(y_test)[1]
    predicted_player_value = y_pred[1]
    print('True value for the first player in the test set in millions is : ' + str(true_player_value/10**6))
    print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value/10**6))
    print("Accuracy:", r2_score(y_test, y_pred))

    pickle.dump(model, open('XGBmodel.pkl', 'wb'))  # save the trained model




