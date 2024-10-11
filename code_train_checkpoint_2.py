import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler 
from sklearn.tree import DecisionTreeClassifier
import pickle
#import RandomUnderSampler
#data import
df= pd.read_csv('Financial_inclusion_dataset.csv')

# Data splitting
X=df.drop(["bank_account"],axis=1)
y=df["bank_account"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#rus=RandomUnderSampler()
#X_train,y_train=rus.fit_resample(X_train,y_train)


# Initialization (création des instances)
rs = RobustScaler()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
le7 = LabelEncoder()
le8 = LabelEncoder()
le9 = LabelEncoder()
le10 = LabelEncoder()


def fit_transform(df,rs,le1,le2,le3,le4,le5,le6,le7,le8,le9,le10):
    # Fit the imputer and transform tnumerical columns to normalization
    df.loc[:,["household_size",	"age_of_respondent"]] = rs.fit_transform(df[["household_size",	"age_of_respondent"]])
    
    # Encoding cotegorical variables
    le1.fit(df["country"])
    le2.fit(df["uniqueid"])
    le3.fit(df["bank_account"])
    le4.fit(df["location_type"])
    le5.fit(df["cellphone_access"])
    le6.fit(df["gender_of_respondent"])
    le7.fit(df["relationship_with_head"])
    le8.fit(df["marital_status"])
    le9.fit(df["education_level"])
    le10.fit(df["job_type"])

    df["country"]=le1.transform(df["country"])
    df["uniqueid"]=le2.transform(df["uniqueid"])
    df["bank_account"]=le3.transform(df["bank_account"])
    df["location_type"]=le4.transform(df["location_type"])
    df["cellphone_access"]=le5.transform(df["cellphone_access"])
    df["gender_of_respondent"]=le6.transform(df["gender_of_respondent"])
    df["relationship_with_head"]=le7.transform(df["relationship_with_head"])
    df["marital_status"]=le8.transform(df["marital_status"])
    df["education_level"]=le9.transform(df["education_level"])
    df["job_type"]=le10.transform(df["job_type"])

fit_transform(X_train,rs,le1,le2,le3,le4,le5,le6,le7,le8,le9,le10)


###Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Serialization
data={"rs":rs,"le1":le1,"le2":le2,"le3":le3,"le4":le4,"le5":le5,"le6":le6,"le7":le7,"le8":le8,"le9":le9,"le10":le10,"model":model}
filename = "data.pickle"
#step 2:  creation (nom fichier avec extension pickle , write binary)
with open(filename, 'wb') as file: 
    #step 1: encodage byte en byte stream
    pickle.dump(data, file)
