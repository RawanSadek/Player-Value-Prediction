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
1.	XGBoost: XGBoost is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.
2.	Polynomial Regression: Polynomial Regression a form of Linear regression known as a special case of Multiple linear regression which estimates the relationship as an nth degree polynomial. Polynomial Regression is sensitive to outliers so the presence of one or two outliers can also badly affect the performance.

## Differences Between Each Model and The Acquired Results:
### 1.	XGBoost:
●	Accuracy: 0.9951267345309258 (99%)
●	MSE: 145785361670.41998
●	Training Time: about 4.5  sec
### 2.	Polynomial Regression:
●	Accuracy: 0.9863949342990068 (98%)
●	MSE: 461662358925.458
●	Training Time: about 0.3 sec.

