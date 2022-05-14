# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from sklearn.preprocessing import RobustScaler

from fastapi import FastAPI
from BankNotes import Malade
import numpy as np
import pickle
import pandas as pd
from diabetes import Diabetes
from fastapi.middleware.cors import CORSMiddleware

from recommander import Recommender
# 2. Create the app object
app = FastAPI()



@app.post('/predict_diet')
def diet(user_id):
    user_id = 'User_'+user_id  # user id of current user

    profiles = pd.read_csv('user_Profiles.csv') # profiles of all users
    recent_activity = pd.read_csv('recent_activity.csv') # recent activities of current user (meals liked,rated,searched,Purchased)
    dataset = pd.read_csv('dataset.csv') # main dataset
    ob = Recommender(profiles,recent_activity,dataset)
    result = ob.recommend(user_id)
    result
    return {'message': result}

@app.post('/predict_symptoms')
def symptoms(data:Symptoms):
    data = data.dict()
    sym1=data['sym1']
    sym2=data['sym2']
    sym3=data['sym3']
    sym4=data['sym4']
    sym5=data['sym5']
    sym6=data['sym6']
    sym7=data['sym7']
    sym8=data['sym8']
    sym9=data['sym9']
    sym10=data['sym10']
    sym11=data['sym11']
    sym12=data['sym12']
    sym13=data['sym13']
    sym14=data['sym14']
    sym15=data['sym15']
    sym16=data['sym16']
    sym17=data['sym17']


    vals  = np.array([sym1,sym2,sym3,sym4,sym5,sym6,sym7,sym8,sym9,sym10,sym11,sym12,sym13,sym14,sym15,sym16,sym17])

    df_severity = pd.read_csv('Symptom-severity.csv')
    df_severity['Symptom'] = df_severity['Symptom'].str.replace('_',' ')
    symptoms = df_severity['Symptom'].unique()
    vals =  np.where(vals =='nan', 0, vals)
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df_severity[df_severity['Symptom'] == symptoms[i]]['weight'].values[0]
    vals = list(map(int, vals))
    print(vals)
    result = symptoms_model.predict([vals])
    return {'resultat': result[0]}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict_heart')
def predict_banknote(data:Malade):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data = data.dict()
    Age=data['Age']
    RestingBP=data['RestingBP']
    Cholesterol=data['Cholesterol']
    FastingBS=data['FastingBS']
    MaxHR=data['MaxHR']
    Oldpeak=data['Oldpeak']
    Sex=data['Sex']
    ChestPainType=data['ChestPainType']
    RestingECG=data['RestingECG']
    ExerciseAngina=data['ExerciseAngina']
    ST_Slope=data['ST_Slope']

    arr = [data['Age'],data['Sex'],data['ChestPainType'],data['RestingBP'],data['Cholesterol'],data['FastingBS'],data['RestingECG'],
        MaxHR, ExerciseAngina, Oldpeak, ST_Slope]
    df = pd.DataFrame([arr] , columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
       'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope',]) 

    numerical_features = df.drop(['Sex', 'ChestPainType', 'RestingBP' , 'RestingECG' , 'ExerciseAngina' , 'ST_Slope'], axis = 1)
    numerical_features = numerical_features.apply(LabelEncoder().fit_transform)

    categorical_features = df.drop(['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak'], axis = 1)
    categorical_features = pd.get_dummies(categorical_features)

    X = pd.concat([numerical_features, categorical_features], axis=1)
    all_columns  = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
       'HeartDisease', 'Sex_F', 'Sex_M', 'ChestPainType_ASY',
       'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
       'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
       'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down',
       'ST_Slope_Flat', 'ST_Slope_Up']
    diff = np.setdiff1d(all_columns ,X.columns)   
 
    for col in diff:
        X[col] = 0
    #X = X.to_numpy()
    #print(arr)
    #print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict_proba(X)
    
    return {
        'pourcentage': prediction[0,0]
    }
  
        

@app.post('/predict_diet')
def diet(user_id):
    user_id = 'User_'+user_id  # user id of current user

    profiles = pd.read_csv('user_Profiles.csv') # profiles of all users
    recent_activity = pd.read_csv('recent_activity.csv') # recent activities of current user (meals liked,rated,searched,Purchased)
    dataset = pd.read_csv('dataset.csv') # main dataset
    ob = Recommender(profiles,recent_activity,dataset)
    result = ob.recommend(user_id)
    result
    return {'message': result}



def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"     

pickle_in = open("heart-failure-model.pkl","rb")
pickle1 = open("symptoms.pkl","rb")
pickle2 = open("newdiabetes.pkl","rb")


classifier=pickle.load(pickle_in)
symptoms_model=pickle.load(pickle1)
diabetes_model=pickle.load(pickle2)



# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post('/predict_diabetes')
def diabetes(data:Diabetes):
    data = data.dict()
    Pregnancies=data['Pregnancies']
    Glucose=data['Glucose']
    BloodPressure=data['BloodPressure']
    SkinThickness=data['SkinThickness']
    Insulin=data['Insulin']
    BMI=data['BMI']
    DiabetesPedigreeFunction=data['DiabetesPedigreeFunction']
    Age=data['Age']


    arr = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    df = pd.DataFrame([arr] , columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']) 


    # According to BMI, some ranges were determined and categorical variables were assigned.
    NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")

    df["NewBMI"] = NewBMI

    df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]

    df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
    df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]

    df["NewInsulinScore"] = df.apply(set_insulin, axis=1)
    NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")

    df["NewGlucose"] = NewGlucose

    df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]

    df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]

    df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]

    df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]
    df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)

    all_columns  = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'NewBMI_Obesity 1',
       'NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight',
       'NewBMI_Underweight', 'NewInsulinScore_Normal', 'NewGlucose_Low',
       'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']
    diff = np.setdiff1d(all_columns ,df.columns)  
    for col in diff:
        df[col] = 0

    categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]

    X = df.drop(['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)
    cols = X.columns
    index = X.index 
    print(cols)
    print(index)
 

    X = (X - np.nanmean(X, dtype = 'float32')) / np.nanstd(X, dtype = 'float32')


    X = pd.concat([X, categorical_df], axis = 1)  
    prediction = diabetes_model.predict_proba(X)
    
    print(X)
    return {
        'pourcentage': prediction[0,0]
    }
              




origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers=["*"]
)





# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload