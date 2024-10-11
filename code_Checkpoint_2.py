#Import modules
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

#user welcoming
st.write("Welcom on board")
st.title("You can provide the required features and predict which individuals are most likely to have or use a bank account.")

#user inputs 
# Liste des pays d'Afrique
african_countries = ["Afrique du Sud", "Algérie", "Angola", "Bénin", "Botswana", 
    "Burkina Faso", "Burundi", "Cabo Verde", "Cameroun", 
    "République centrafricaine", "Tchad", "Comores", 
    "République du Congo", "République démocratique du Congo", 
    "Djibouti", "Égypte", "Équateur", "Érythrée", "Eswatini", 
    "Éthiopie", "Gabon", "Gambie", "Ghana", "Guinée", 
    "Guinée-Bissau", "Côte d'Ivoire", "Kenya", "Lesotho", 
    "Libéria", "Libye", "Madagascar", "Malawi", "Mali", 
    "Maroc", "Mauritanie", "Maurice", "Mozambique", "Namibie", 
    "Niger", "Nigeria", "Rwanda", "Sao Tomé-et-Principe", 
    "Sénégal", "Seychelles", "Sierra Leone", "Somalie", 
    "Soudan", "Soudan du Sud", "Tanzanie", "Togo", "Tunisie", 
    "Uganda", "Zambie", "Zimbabwe"]

# Titre de l'application
st.title("Sélectionnez un pays d'Afrique")
# Liste déroulante pour sélectionner un pays
selected_country = st.selectbox("Choisissez un pays :", african_countries)

# Afficher le pays sélectionné
country= st.write(f"Vous avez sélectionné : {selected_country}")
uniqueid= st.slider("Please provide the ID number of the client", 1, 10000, 0, 1)      
location_type =st.selectbox("Choose the client's location type",["urban", "rural"])  
cellphone_access =     st.selectbox("Does the client has access to cellphone?",("Yes", "No"),)     
household_size   =     st.number_input("Please provide the rooms number of the client's house", min_value=0, step=1, format="%d")       
age_of_respondent =    st.slider("Please provide the age of the client", 1, 80, 0, 1)  
gender_of_respondent =    st.selectbox("What is the gender of the client?",("Male", "Female"),)  


relationship_with_head=   st.selectbox("relationship_with_head?",['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives'],)    
marital_status  =     st.selectbox("What is the marital status?",['Married/Living together', 'Widowed', 'Single/Never Married',
       'Divorced/Seperated', 'Dont know'],)      
education_level =   st.selectbox("Please provide his education level?",('Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'),)        
job_type   = st.selectbox("What type of job does the client have?",('Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'),) 
 


# put features in DataFrame
user_data=pd.DataFrame({'country':[country],'uniqueid':[uniqueid],'Dlocation_type':[location_type],
                        'cellphone_access':[cellphone_access], 'household_size':[household_size],
                        'age_of_respondent':[age_of_respondent],
                        'gender_of_respondent':[gender_of_respondent],
                        'relationship_with_head':[relationship_with_head],
                        'marital_status':[marital_status],
                        'education_level':[education_level],
                        'job_type':[job_type],})


# Initialization (création des instances)
with open('memory.pkl', 'rb') as f: 
    objects_list=pickle.load(f)

rs = objects_list[0]
le1 = objects_list[1]
le2 = objects_list[2]
le3 = objects_list[3]
le4 = objects_list[4]
le5= objects_list[5]
le6= objects_list[6]
le7= objects_list[7]
le8= objects_list[8]
le9= objects_list[9]
le10= objects_list[10]

def transform(user_data,rs,le1,le2,le3,le4,le5,le6,le7,le8,le9,le10):
    # transform numerical columns to normalization
    user_data.loc[:,["household_size",	"age_of_respondent"]] = rs.transform(user_data[["household_size",	"age_of_respondent"]])
    
    user_data["country"]=le1.transform(user_data["country"])
    user_data["uniqueid"]=le2.transform(user_data["uniqueid"])
    user_data["bank_account"]=le3.transform(user_data["bank_account"])
    user_data["location_type"]=le4.transform(user_data["location_type"])
    user_data["cellphone_access"]=le5.transform(user_data["cellphone_access"])
    user_data["gender_of_respondent"]=le6.transform(user_data["gender_of_respondent"])
    user_data["relationship_with_head"]=le7.transform(user_data["relationship_with_head"])
    user_data["marital_status"]=le8.transform(user_data["marital_status"])
    user_data["education_level"]=le9.transform(user_data["education_level"])
    user_data["job_type"]=le10.transform(user_data["job_type"])

#  User input transformation only (without fit)
transform(user_data,rs,le1,le2,le3,le4,le5,le6,le7,le8,le9,le10)

###Decision Tree
model = objects_list[-1]

#  predictions
y__user_pred = model.predict(user_data)
#affichage du résultat
if y__user_pred ==1:
    st.subheader("You are expected to be alive")
else :
    st.subheader("You are expected to be dead")




