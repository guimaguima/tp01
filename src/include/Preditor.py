from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import random
import time

class BasePredictor(ABC):
    
    @abstractmethod
    def predict(self, patient_data: dict) -> str:
        pass
    
    @abstractmethod
    def predict_proba(self, patient_data: dict) -> dict:
        pass


class MockPredictor(BasePredictor):
    def __init__(self):
        self.classes = ['NO', '<30', '>30']

    def predict_proba(self, patient_data: dict) -> dict:
        time.sleep(1.5) 
        
        time_in_hosp = patient_data.get('time_in_hospital', 1)
        
        if time_in_hosp > 7:
            probs = {'NO': 0.20, '<30': 0.50, '>30': 0.30}
        else:
            probs = {'NO': 0.70, '<30': 0.10, '>30': 0.20}
            
        return probs

    def predict(self, patient_data: dict) -> str:
        probs = self.predict_proba(patient_data)
        return max(probs, key=probs.get)