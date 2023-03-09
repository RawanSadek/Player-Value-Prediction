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
