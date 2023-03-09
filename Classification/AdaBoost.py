import os
from sklearn.model_selection import train_test_split
from pre_processing import *
import pickle
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings('ignore')
warnings.warn('ConvergenceWarning')

if os.path.exists("ABCmodel.pkl"):
    testData = pd.read_csv("player-test-samples.csv")
    pd.set_option('display.max_rows', None)
    x = testData.iloc[:, :-1]
    # actual y
    y = testData["PlayerLevel"]
    x, y = preProcessing(testData, x, y)

    model = pickle.load(open('ABCmodel.pkl', 'rb'))  # load the trained model   #############
    y_pred = model.predict(x)

    accuracy = np.mean(y_pred == y) * 100
    print("The achieved accuracy using Adaboost is " + str(accuracy))
    true_player_value = np.asarray(y)[0]
    predicted_player_value = y_pred[0]
    print('True value for the first player is : ' + str(true_player_value))
    print('Predicted value for the first player is : ' + str(predicted_player_value))

else:
    data = pd.read_csv("player-classification.csv")
    pd.set_option('display.max_rows', None)
    X = data.iloc[:, :-1]
    # actual y
    Y = data["PlayerLevel"]
    X, Y = preProcessing(data, X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

    ABC = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=100,learning_rate=0.2)

    ABC.fit(X_train, y_train)
    y_prediction = ABC.predict(X_test)
    accuracy = np.mean(y_prediction == y_test) * 100
    print("The achieved accuracy using Adaboost is " + str(accuracy))
    true_player_value = np.asarray(y_test)[0]
    predicted_player_value = y_prediction[0]
    print('True value for the first player is : ' + str(true_player_value))
    print('Predicted value for the first player is : ' + str(predicted_player_value))

    pickle.dump(ABC, open('ABCmodel.pkl', 'wb'))  # save the trained model   #############

