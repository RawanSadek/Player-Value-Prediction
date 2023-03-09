from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
import pandas as pd
from sklearn.impute import SimpleImputer


def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

def date_preprocessing(x):
    if x is np.nan:
        pass
    else:
       x = str(x)
       x = x[-2:]
       return x

#-------------------------------------------------------Pre-Processing-------------------------------------------------------------

def preProcessing(data,X,Y):
    # Dealing with uncommon/exceptional cells in body type feature
    # Series.value_counts(normalize=False, sort=True, ascending=False, bins=None(optional, works only with numeric data), dropna=True)
    lst = []
    count = 0
    for j in X['body_type']:
        if j == "Normal" or j == "Stocky" or j == "Lean":
            lst.append(j)
            count += count
        else:
            BMI = int(X["weight_kgs"][count]) / (0.01 * int((X["height_cm"][count]) ** 2))
            if BMI >= 25:
                lst.append("Stocky")
                count += count
            elif BMI <= 18.5:
                lst.append("Lean")
                count += count
            else:
                lst.append("Normal")
                count += count
    X['body_type'] = lst

    X['Lean_body_type'] = 0
    X['Normal_body_type'] = 0
    X['Stocky_body_type'] = 0
    l = []
    n = []
    s = []
    for i in X['preferred_foot']:
        if i == "Lean":
            l.append(1)
            n.append(0)
            s.append(0)
        elif i == "Normal":
            l.append(0)
            n.append(1)
            s.append(0)
        else:
            l.append(0)
            n.append(0)
            s.append(1)
    X['Lean_body_type'] = l
    X['Normal_body_type'] = n
    X['Stocky_body_type'] = s


    X['Right_preferred_foot'] = 0
    X['Left_preferred_foot'] = 0
    l=[]
    R=[]
    for i in X['preferred_foot']:
        if i=="Right":
            l.append(0)
            R.append(1)
        elif i=="Left":
            l.append(1)
            R.append(0)
        else:
            l.append(0)
            R.append(0)
    X['Right_preferred_foot']=R
    X['Left_preferred_foot']=l

    X.drop(['body_type','preferred_foot'], axis = 1,inplace=True)
    # converts categorical data into indicator variables
    #X = pd.get_dummies(X, columns=['body_type', 'preferred_foot'], drop_first=False)

    # converts ordinal categorical data into numeric data
    X[['work_rate_attacking', 'work_rate_defense']] = X['work_rate'].str.split('/', expand=True)
    workrate1 = {"Low": 1, "Medium": 2, "High": 3}
    X["work_rate_attacking"] = X["work_rate_attacking"].replace(workrate1)
    workrate2 = {" Low": 1, " Medium": 2, " High": 3}
    X["work_rate_defense"] = X["work_rate_defense"].replace(workrate2)

    X.drop(['work_rate'], axis=1, inplace=True)

    # preprocess contract_end_year column
    X['contract_end_year'] = X['contract_end_year'].apply(date_preprocessing)

    # preprocess position power columns (adding the two values)
    position_power = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
                      'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

    for i in position_power:
        X[i] = X[i].fillna("0+0")

    l = []
    for col in position_power:
        for index in range(len(X)):
            values = X[col][index].split('+')
            l.append(int(values[0]) + int(values[1]))

        X[col] = l
        l = []

    # encoding (columns to be dropped)
    cols = ("name", "full_name", "nationality", "birth_date", "club_join_date", "club_team")
    X = Feature_Encoder(X, cols)

    # updating columns with boolean values (0,1)
    ll = []
    for i in X['national_rating']:
        if math.isnan(i):
            ll.append(0)
        else:
            ll.append(1)
    X['national_team'] = ll
    # X['national_rating']=ll
    X['national_rating'] = X["national_rating"].fillna(0)

    cols = ['traits', 'tags']
    ll = []
    for i in cols:
        for j in X[i]:
            if j is np.nan:
                cnt = 0
            else:
                values = str(j).split(',')
                cnt = len(values)
            ll.append(cnt)
        X[i] = ll
        ll = []
    # converts ordinal categorical data into numeric data
    attaker = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']
    midline = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB']
    defender = ['LB', 'LCB', 'CB', 'RCB', 'RB']
    sub = ['SUB', 'RES']
    values_list = []
    for i in X["positions"]:
        value = 0
        lst = []
        lst = i.split(',')
        for j in range(len(lst)):
            if lst[j] in sub:
                value += 1
            elif lst[j] == 'GK':
                value += 2
            elif lst[j] in defender:
                value += 3
            elif lst[j] in midline:
                value += 4
            elif lst[j] in attaker:
                value += 5
        values_list.append(value)
    X["positions"] = values_list

    X["national_team_position"] = X["national_team_position"].fillna("0")
    cols = ["club_position", "national_team_position"]
    value = 0
    for i in cols:
        values_list = []

        for j in X[i]:
            if j in sub:
                value += 1
            elif j == 'GK':
                value += 2
            elif j in defender:
                value += 3
            elif j in midline:
                value += 4
            elif j in attaker:
                value += 5
            else:
                value = 0
            values_list.append(value)
            value = 0
        X[i] = values_list

    # dealing with Nulls
    col_with_nulls = ["wage", "club_rating", "club_jersey_number", "club_join_date", "contract_end_year",
                      "release_clause_euro", "national_jersey_number"]
    imputer = SimpleImputer(strategy='median')
    for col in col_with_nulls:
        X[col] = X[col].fillna(X[col].median()).astype(float)

    # scaling
    X_cols = ["id", "wage", "release_clause_euro"]
    X[X_cols] = featureScaling(X[X_cols], 0, 1)

    X.to_csv("new fifa classification.csv", index=False)
    X_total = pd.read_csv("new fifa classification.csv")


    # Feature Selection
    X.drop(['id', 'name', 'full_name', 'birth_date', 'height_cm', 'weight_kgs', 'nationality',
            'club_team', 'club_position', 'club_jersey_number', 'club_join_date', 'national_team_position',
            'national_team','national_jersey_number', 'tags', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
            'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB','sliding_tackle', 'GK_diving', 'GK_handling', 'GK_kicking',
            'GK_positioning', 'GK_reflexes', 'Lean_body_type', 'Normal_body_type', 'Stocky_body_type',
            'Left_preferred_foot', 'Right_preferred_foot','work_rate_defense','positions', 'weak_foot(1-5)',
            'heading_accuracy', 'acceleration', 'sprint_speed', 'agility', 'balance','jumping', 'strength', 'aggression',
            'interceptions', 'marking', 'standing_tackle', 'work_rate_attacking', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'LS',
            'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW'], axis=1,
           inplace=True)


    # dealing with Nulls in PlayerLevel column
    X_total.dropna(how='any', inplace=True)
    X_total.to_csv("new fifa classification.csv", index=False)

    return X,Y


