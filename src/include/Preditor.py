from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
from pickle_blosc import pickle, unpickle
import time

class BasePredictor(ABC):
    
    @abstractmethod
    def predict(self, patient_data: dict) -> str:
        pass
    
    @abstractmethod
    def predict_proba(self, patient_data: dict) -> dict:
        pass


class RandomForestPredictor(BasePredictor):
    def __init__(self):
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pickled', 'randomforest_pipeline.pkl')
        )

        self.classes = ['NO', '<30', '>30']
        self.encoded_classes = ['<30', '>30', 'NO']
        self.pipeline = self.__load_model(model_path)

        classifier = self.pipeline.named_steps.get('classifier')
        if classifier is not None and hasattr(classifier, 'n_jobs'):
            classifier.n_jobs = 1

    def __load_model(self, model_path: str):
        return unpickle(model_path)

    def __prepare_input(self, patient_data: dict) -> pd.DataFrame:
        feature_names = list(self.pipeline.feature_names_in_)
        row = {feature: patient_data.get(feature, np.nan) for feature in feature_names}

        # Trata "None" como nan
        for column in ['A1Cresult', 'max_glu_serum']:
            if row.get(column) == 'None':
                row[column] = np.nan

        return pd.DataFrame([row], columns=feature_names)

    def predict_proba(self, patient_data: dict) -> dict:
        patient_df = self.__prepare_input(patient_data)
        probabilities = self.pipeline.predict_proba(patient_df)[0]

        encoded_probability_map = {
            self.encoded_classes[int(encoded_class)]: float(probability)
            for encoded_class, probability in zip(self.pipeline.classes_, probabilities)
        }

        return {
            label: encoded_probability_map.get(label, 0.0)
            for label in self.classes
        }

    def predict(self, patient_data: dict) -> str:
        patient_df = self.__prepare_input(patient_data)
        encoded_prediction = int(self.pipeline.predict(patient_df)[0])

        return self.encoded_classes[encoded_prediction]
