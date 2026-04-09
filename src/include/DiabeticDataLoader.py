import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
class DiabeticDataLoader:
    def __init__(self, df_raw):
        self.df_raw = df_raw.copy()
        self.df_clean = None
        self.df_no_outliers = None
        self.scaler = None

    def clean_data(self):
        df = self.df_raw.copy()
        
        # Remover colunas constantes
        constant_cols = ['examide', 'citoglipton']
        df.drop(columns=constant_cols, inplace=True, errors='ignore')
        
        # Substituir '?' por NaN de todas as colunas
        df.replace('?', np.nan, inplace=True)
        
        # Lidar com max_glu_serum e A1Cresult
        for col in ['max_glu_serum', 'A1Cresult']:
            df[col] = df[col].replace('None', np.nan)
            df[col + '_measured'] = (~df[col].isna()).astype(int)
            
        # Indicador de peso registrado e conversão
        df['weight_recorded'] = (~df['weight'].isna()).astype(int)
        df['weight'] = df['weight'].astype('category')
        
        # Colunas categóricas
        categorical_cols = [
            'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'payer_code', 'medical_specialty',
            'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
            'change', 'diabetesMed', 'readmitted'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        self.df_clean = df
        return df

    def remove_outliers(self):
        if self.df_clean is None:
            raise ValueError("Data must be cleaned before outlier removal.")
        
        df = self.df_clean.copy()

        outlier_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient'
        ]
        
        self.thresholds = {}
        for col in outlier_cols:
            if col in df.columns:
                # Calcula Min e P99 para ser o thresholds
                low = df[col].min()
                high = df[col].quantile(0.99)
                self.thresholds[col] = (low, high)
        
        mask = pd.Series(True, index=df.index)
        for feature, (low, high) in self.thresholds.items():
            if feature in df.columns:
                mask &= df[feature].between(low, high)
                
        df_no_outliers = df.loc[mask].copy()
        self.df_no_outliers = df_no_outliers
        return df_no_outliers
    
    def get_clean_data(self):
        if self.df_clean is None:
            self.clean_data()
        return self.df_clean
    
    def get_no_outliers_data(self):
        if self.df_no_outliers is None:
            self.remove_outliers()
        return self.df_no_outliers

    def get_features_target(self):
        df = self.get_no_outliers_data()
        exclude_cols = ['encounter_id', 'patient_nbr', 'readmitted']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df['readmitted']
        return X, y

    def get_train_test_split(self, test_size=0.2, random_state=42):
        # Limpa os dados de forma global
        df_clean = self.clean_data()
        
        # Separa X e Y
        X, y = self.get_features_target(df_clean)
        
        # Faz o split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Remove outliers no train
        train_combined = pd.concat([X_train, y_train], axis=1)
        train_combined = self.remove_outliers(train_combined, calculate_thresholds=True)
        X_train = train_combined.drop(columns=['readmitted'])
        y_train = train_combined['readmitted']
        
        # Padronização
        numeric_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'num_procedures', 'number_diagnoses'
        ]
        
        self.scaler = StandardScaler()
        # FIT apenas no treino
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        # TRANSFORM no teste
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, directory=os.path.join('..','data', 'processed')):
        if self.df_no_outliers is None:
            raise ValueError("No processed data to save. Please run the processing steps first.")
        
        os.makedirs(directory, exist_ok=True)
        self.df_no_outliers.to_csv(os.path.join(directory, 'diabetic_data_processed.csv'), index=False)