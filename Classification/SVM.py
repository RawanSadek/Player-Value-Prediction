import os
import time
from sklearn import svm, metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pre_processing import *
import pickle
import warnings

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

if os.path.exists("SVMmodel.pkl"):
    testData = pd.read_csv("player-test-samples.csv")
    removenull = pd.notnull(testData)
    testData[removenull]
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    # actual y
    y = testData["PlayerLevel"]
    x, y = preProcessing(testData, x, y)

    model = pickle.load(open('SVMmodel.pkl', 'rb'))  # load the trained model   #############
    #model.predict(x)
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

    model = svm.SVC(kernel='linear', C=3).fit(X_train, y_train)

    # Training Time
    start = time.time()
    model.fit(X_train, y_train)
    stop = time.time()
    print("training time = ", stop - start, " sec.")

    y_pred = model.predict(X_test)
    # model accuracy for svc model
    print('SVC Kernel Function')
    print("===================")
    accuracy = model.score(X_test, y_test)
    print('test accuracy: ', accuracy * 100)
    accuracy = model.score(X_train, y_train)
    print('training accuracy: ', accuracy * 100)

    pickle.dump(model, open('SVMmodel.pkl', 'wb'))  # save the trained model   #############



