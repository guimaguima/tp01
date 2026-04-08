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
        self.df_treated = None
        

    def clean_data(self):
        df = self.df_raw.copy()
        
        # Drop constant columns
        constant_cols = ['examide', 'citoglipton']
        df.drop(columns=constant_cols, inplace=True, errors='ignore')
        
        # Replace '?' with NaN in weight
        df['weight'] = df['weight'].replace('?', np.nan)
        
        # Handle max_glu_serum and A1Cresult: 'None' means test not performed
        for col in ['max_glu_serum', 'A1Cresult']:
            df[col] = df[col].replace('None', np.nan)
            df[col + '_measured'] = (~df[col].isna()).astype(int)
            
        # Create weight recorded indicator
        df['weight_recorded'] = (~df['weight'].isna()).astype(int)
        
        # Convert weight to categorical (keep original bins)
        df['weight'] = df['weight'].astype('category')
        
        # Convert categorical columns to category dtype
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
    
    def treat_outliers(self):
        if self.df_clean is None:
            raise ValueError("Data must be cleaned before outlier treatment.")
        
        df = self.df_clean.copy()
        
        # Define capping thresholds and features to treat
        outlier_rules = {
            'time_in_hospital': (1, 14),
            'num_lab_procedures': (0, 85),
            'num_medications': (0, 60),
            'number_outpatient': (0, 20),
            'number_emergency': (0, 10),
            'number_inpatient': (0, 10)
        }
        
        # Apply capping and create outlier flags
        for feature, (lower, upper) in outlier_rules.items():
            if feature in df.columns:
                flag_col = feature + '_outlier_flag'
                df[flag_col] = ((df[feature] < lower) | (df[feature] > upper)).astype(int)
                df[feature] = df[feature].clip(lower=lower, upper=upper)
        
        # Features left unchanged (e.g., num_procedures, number_diagnoses)
        self.df_treated = df
        return df

    def remove_outliers(self):
        if self.df_clean is None:
            raise ValueError("Data must be cleaned before outlier removal.")
        
        df = self.df_clean.copy()
        
        # Define outlier thresholds
        thresholds = {
            'time_in_hospital': (1, 14),
            'num_lab_procedures': (0, 85),
            'num_medications': (0, 60),
            'number_outpatient': (0, 20),
            'number_emergency': (0, 10),
            'number_inpatient': (0, 10)
        }
        
        # Create boolean mask for rows without outliers
        mask = pd.Series(True, index=df.index)
        for feature, (low, high) in thresholds.items():
            if feature in df.columns:
                mask &= df[feature].between(low, high)
                
        # Filter out rows with outliers
        df_no_outliers = df.loc[mask].copy()
        self.df_no_outliers = df_no_outliers
        return df_no_outliers

    def standardize_features(self):
        if self.df_no_outliers is None:
            raise ValueError("Outliers must be removed before standardization.")
            
        df = self.df_no_outliers.copy()
        numeric_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'num_procedures', 'number_diagnoses'
        ]
        
        # Initialize scaler and fit-transform numeric features
        self.scaler = StandardScaler()
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        self.df_no_outliers = df
        return df

    def get_clean_data(self):
        if self.df_clean is None:
            self.clean_data()
        return self.df_clean
    
    def get_outlier_treated_data(self):
        if self.df_treated is None:
            self.treat_outliers()
        return self.df_treated

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
        X, y = self.get_features_target()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        return X_train, X_test, y_train, y_test