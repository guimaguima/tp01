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

    def _map_icd9(self, val):
        """Mapeia os códigos CID-9 para as 9 macro categorias presentes no artigo sobre esse Dataset."""
        if pd.isna(val) or val == '?':
            return 'Missing'

        val_str = str(val)
        # Códigos suplementares (V e E) entram em 'Other'
        if val_str.startswith('V') or val_str.startswith('E'):
            return 'Other'

        try:
            # Pega apenas a raiz do diagnóstico antes do ponto
            num = float(val_str.split('.')[0])

            if num == 250:
                return 'Diabetes'
            elif (390 <= num <= 459) or num == 785:
                return 'Circulatory'
            elif (460 <= num <= 519) or num == 786:
                return 'Respiratory'
            elif (520 <= num <= 579) or num == 787:
                return 'Digestive'
            elif 800 <= num <= 999:
                return 'Injury'
            elif 710 <= num <= 739:
                return 'Musculoskeletal'
            elif (580 <= num <= 629) or num == 788:
                return 'Genitourinary'
            elif 140 <= num <= 239:
                return 'Neoplasms'
            else:
                return 'Other'
        except ValueError:
            return 'Other'

    def clean_data(self):
        df = self.df_raw.copy()

        # 1. Remover colunas desnecessárias, com alta porcentagem de vazios, ou constantes avaliados pelo artigo
        constant_cols = ['examide', 'citoglipton', 'weight', 'payer_code', 'encounter_id', 'patient_nbr', 'weight', 'medical_specialty']
        df.drop(columns=constant_cols, inplace=True, errors='ignore')

        # 2. Tratamento de valores faltantes e marcações nulas
        df.replace('?', np.nan, inplace=True)
        for col in ['max_glu_serum', 'A1Cresult']:
            df[col] = df[col].replace('None', np.nan)

        # 3. Agregação CID-9
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                df[col] = df[col].apply(self._map_icd9)

        # 4. Redução da Dimensionalidade das colunas de medicamentos
        med_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        df['total_meds_up'] = (df[med_cols] == 'Up').sum(axis=1)
        df['total_meds_down'] = (df[med_cols] == 'Down').sum(axis=1)
        df['total_meds_steady'] = (df[med_cols] == 'Steady').sum(axis=1)
        df.drop(columns=med_cols, inplace=True, errors='ignore')

        # 5. Redução da Dimensionalidade das colunas de contagem de visitas anteriores do paciente
        visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        df['total_prior_visits'] = df[visit_cols].sum(axis=1)
        df.drop(columns=visit_cols, inplace=True, errors='ignore')

        # 6. Atualização de tipos para Categóricos
        categorical_cols = [
            'race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'medical_specialty',
            'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult',
            'change', 'diabetesMed', 'readmitted'
        ]
        for col in categorical_cols:
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype('category')

        # DataFrame Limpo e com Dimensão Reduzida
        self.df_clean = df
        return df

    def remove_outliers(self, df=None):
        if df is None:
            if self.df_clean is None:
                raise ValueError("Data must be cleaned before outlier removal.")
            else:
                df = self.df_clean.copy()

        outlier_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'total_prior_visits'
        ]

        self.thresholds = {}
        for col in outlier_cols:
            if col in df.columns:
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
        df = self.get_clean_data()
        exclude_cols = ['encounter_id', 'patient_nbr', 'readmitted']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df['readmitted']
        return X, y

    def get_train_test_split(self, test_size=0.2, random_state=42):
        df_clean = self.clean_data()
        X, y = self.get_features_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        train_combined = pd.concat([X_train, y_train], axis=1)
        train_combined = self.remove_outliers(train_combined)

        X_train = train_combined.drop(columns=['readmitted'])
        y_train = train_combined['readmitted']

        numeric_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_medications',
            'total_prior_visits', 'num_procedures', 'number_diagnoses',
            'total_meds_up', 'total_meds_down', 'total_meds_steady'
        ]

        self.scaler = StandardScaler()
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train, X_test, y_train, y_test