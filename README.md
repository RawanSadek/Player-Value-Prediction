# Player-Value-Prediction
The objective of the projects is to apply different machine learning algorithms to real-world tasks. We will show how to clean data, applying pre-processing, feature scaling, regression, and classification methods. Given the provided datasets, we would like to understand and predict a player’s value/level based on the provided data.
## Preprocessing techniques applied on the dataset: 
1.	Get dummies: used on body_type column to separate the categorical data into 3 columns; body type lean, body type normal, body type stocky.
2.	Replace: replaces string values with numeric values that help describe the degree.
3.	Fillna: replaces all null values with more suitable numeric values.
4.	Date_Preprocessing: replaces all values with the last 2 characters in the original string.
5.	LabelEncoder: converts any string value to a numeric value.
6.	SimpleImputer: replaces null values found in columns with numeric values using a certain strategy (ex: Median). 
7.	FeatureScaling: scales large numeric values.
8.	Body_type Preprocessing: a function to replace null values with a calculated numeric value that represents the body type by using BMI rules.

## Dataset Analysis: 
We used the heat map function to know how each feature affects the other.
![image](https://user-images.githubusercontent.com/76558250/224090818-b25dc0d9-84d7-4ae9-a3e1-480580bc0e5a.png)

## Regression Techniques Used:
### 1.	XGBoost: 
XGBoost is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

### 2.	Polynomial Regression: 
Polynomial Regression a form of Linear regression known as a special case of Multiple linear regression which estimates the relationship as an nth degree polynomial. Polynomial Regression is sensitive to outliers so the presence of one or two outliers can also badly affect the performance.

## Differences Between Each Model and The Acquired Results:
### 1.	XGBoost:
●	Accuracy: 0.9951267345309258 (99%)
●	MSE: 145785361670.41998
●	Training Time: about 4.5  sec
### 2.	Polynomial Regression:
●	Accuracy: 0.9863949342990068 (98%)
●	MSE: 461662358925.458
●	Training Time: about 0.3 sec.

###### 

## Classification Techniques Used:
### 1.	AdaBoost with Decision Tree: 
AdaBoost is an ensemble learning method (also known as “meta-learning”) which was initially created to increase the efficiency of binary classifiers. AdaBoost uses an iterative approach to learn from the mistakes of weak classifiers, and turn them into strong ones.
AdaBoost, short for Adaptive Boosting, is a statistical classification meta-algorithm.

	Hyper Parameters of Decision Tree:
•	Max_depth: 3  92.37730595196658
                       5  95.47511312217195
                       6  95.63974939088061 
                      10  95.85798816568047
the value 5 gives us the highest testing accuracy with the minimum complexity

	Hyper Parameters of Adaboost: 
•	learning_rate: 0.2  95.370692655760521
1	  94.67455621301775
0.1	 95.23146536721198
            The value 0.2 gives the testing accuracy

	Training time  =  2.1542320251464844  sec.

	Testing time  =  0.06086564064025879  sec.

2.	SVM: The idea of SVM is that the algorithm creates a line or a hyperplane which separates the data into classes.

	Hyper Parameters:
•	C: 1   89.87121475809259
2   89.73198746954402
3   90.01044204664113
               The value 3 gives the highest testing accuracy.

•	Kernel Functions:  SVC with linear kernel  90.01044204664113
                                              Linear SVC  43.543334493560735
                                              SVC with rbf kernel  86.42533936651584
                                              SVC with poly kernel  89.3491124260355

	Training time  =  28.94691491127014  sec.

	Testing time  =  0.33500170707702637  sec.


3.	Logistic Regression: one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

	Hyperparameters:
•	Intercept_scaling: 1  72.43299686738601
                                2  75.67003132613992
                                4  77.06230421162547
The value 4 give the testing accuracy

•	C: 4  78.48938391924817
     2  77.75844065436826
     1  77.27114514444831
The value 4 give the testing accuracy


	Training time  =  0.6350951194763184  sec. 

	Testing time  =  0.009612798690795898  sec.

