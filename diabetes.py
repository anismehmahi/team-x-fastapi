# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020

@author: win10
"""
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Diabetes(BaseModel):
    Pregnancies: float	
    Glucose: float	
    BloodPressure: float
    SkinThickness	: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

    
