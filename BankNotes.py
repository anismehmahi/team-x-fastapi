# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:51:19 2020

@author: win10
"""
from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class Malade(BaseModel):
    Age: float
    Sex: str	
    ChestPainType: str	
    RestingBP: float
    Cholesterol	: float
    FastingBS: float
    RestingECG: str
    MaxHR: float
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str
