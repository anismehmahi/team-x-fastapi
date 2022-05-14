
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from collections import Counter


class Recommender:
    
    def __init__(self,profiles,recent_activity,dataset):
        self.df = dataset
        self.profiles = profiles
        self.recent_activity = recent_activity
    
    def get_features(self,dataframe):
        #getting dummies of dataset
        nutrient_dummies = dataframe.Nutrient.str.get_dummies()
        disease_dummies = dataframe.Disease.str.get_dummies(sep=' ')
        diet_dummies = dataframe.Diet.str.get_dummies(sep=' ')
        feature_df = pd.concat([nutrient_dummies,disease_dummies,diet_dummies],axis=1)
     
        return feature_df
    
    def find_neighbors(self,dataframe,features,k):
        features_df = self.get_features(dataframe)
        total_features = features_df.columns  
        d = dict()
        for i in total_features:
            d[i]= 0
        for i in features:
            d[i] = 1
        final_input = list(d.values())
        
        similar_neighbors = self.k_neighbor([final_input],features_df,dataframe,k)
        return similar_neighbors
    
    def k_neighbor(self,inputs,feature_df,dataframe,k):
        
        #initializing model with k neighbors
        model = NearestNeighbors(n_neighbors=k,algorithm='ball_tree')
        
        # fitting model with dataset features
        model.fit(feature_df)
        
        df_results = pd.DataFrame(columns=list(dataframe.columns))
        
        # getting distance and indices for k nearest neighbor
        distnaces , indices = model.kneighbors(inputs)

        for i in list(indices):
            df_results = df_results.append(dataframe.loc[i])

        df_results = df_results.reset_index(drop=True)
        return df_results
    
    def user_based(self,features,user_id):
       
        similar_users = self.find_neighbors(self.profiles,features,10)
        users = list(similar_users.User_Id)
    
        results = self.recent_activity[self.recent_activity.User_Id.isin(users)] #taking acitivies
   
        results = results[results['User_Id']!=user_id] # selecting those which are not reviewed by user
 
        meals = list(results.Meal_Id.unique())
      
        results = self.df[self.df.Meal_Id.isin(meals)]
    
        results = results.filter(['Meal_Id','Name','Nutrient','Veg_Non','description','Price','Review'])

        results = results.drop_duplicates(subset=['Name'])
        results = results.reset_index(drop=True)
        return results
        
    def recent_activity_based(self,user_id):
        recent_df = self.recent_activity[self.recent_activity['User_Id']==user_id]
        meal_ids = list(recent_df.Meal_Id.unique())
        recent_data = self.df[self.df.Meal_Id.isin(meal_ids)][['Nutrient','catagory','Disease','Diet']].reset_index(drop=True)

        disease = []
        diet = []
        for i in range(recent_data.shape[0]):
            for j in recent_data.loc[i,'Disease'].split():
                disease.append(j)
        for i in range(recent_data.shape[0]):
            for j in recent_data.loc[i,'Diet'].split():
                diet.append(j)
                
        value_counts = recent_data.Nutrient.value_counts()
        m = recent_data.Nutrient.value_counts().mean()
        features = list(value_counts[recent_data.Nutrient.value_counts()>m].index)
        a = dict(Counter(disease))
        
        m = np.mean(list(a.values()))
        for i in a.items():
            if i[1]>m:
                features.append(i[0])
        a = dict(Counter(diet))
        m = np.mean(list(a.values()))
        for i in a.items():
            if i[1]>m:
                features.append(i[0])
                
        similar_neighbors = self.find_neighbors(self.df,features,10)
        return similar_neighbors.filter(['Meal_Id','Name','Nutrient','Veg_Non','description','Price','Review'])
        
    def recommend(self,user_id):
        #finding user's profile features by id
        profile = self.profiles[self.profiles['User_Id']==user_id]
        
        features = []
        features.append(profile['Nutrient'].values[0])
        features.extend(profile['Disease'].values[0].split())
        features.extend(profile['Diet'].values[0].split())
        print(features)
        df1 = self.user_based(features,user_id)
        df2 = self.recent_activity_based(user_id)
        df = pd.concat([df1,df2])
      
        df = df.drop_duplicates('description').reset_index(drop=True)
        return df