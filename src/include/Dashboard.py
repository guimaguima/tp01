import streamlit as st
import pandas as pd
from include.DiabeticDataLoader import DiabeticDataLoader
from include.Plot import Plot
from include.Preditor import BasePredictor, RandomForestPredictor

class Dashboard:
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df = self._load_and_prep_data()
        self.eda_plots = []  
        self.pattern_plots = []
        self.predictor = RandomForestPredictor()

    @st.cache_data 
    def _load_and_prep_data(_self) -> pd.DataFrame:
        df_raw = pd.read_csv(_self.raw_data_path)
        
        loader = DiabeticDataLoader(df_raw)
        
        df_clean = loader.get_clean_data() 
        df_no_outliers = loader.get_no_outliers_data() 
        return df_no_outliers

    def add_eda_plot(self, plot_class: type[Plot]):
        self.eda_plots.append(plot_class(self.df))

    def add_pattern_plot(self, plot_class: type[Plot]):
        self.pattern_plots.append(plot_class(self.df))

    def render_sidebar(self):
        st.sidebar.header("Browse Pages")
        self.page = st.sidebar.radio(
            "Select the page:", 
            [
                "Major Stats (EDA)", 
                "Frequent Pattern for readmission (Mining)",
                "Predictor (ML Model)"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("INFO")
        st.sidebar.info(f"Total Patients Analyzed: {len(self.df)}")


    def render_prediction_page(self):
        st.title("🏥 Readmission Risk Assessment")
        st.markdown("Enter the patient's clinical data to predict the readmission risk.")
        
        if not self.predictor:
            st.error("No AI model has been loaded into the system.")
            return

        with st.form("patient_data_form"):
            st.subheader("Patient Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.selectbox("Age", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
                gender = st.selectbox("Gender", ["Male", "Female"])
                admission_source = st.selectbox("Admission Source (ID)", [1, 2, 3, 7, 17]) # Examples
            
            with col2:
                time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=14, value=3)
                num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=15)
                num_lab_procedures = st.number_input("Lab Procedures", min_value=1, max_value=150, value=40)
            
            with col3:
                a1c_result = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
                diabetes_med = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
                change = st.selectbox("Change in Medication?", ["Ch", "No"])

            submitted = st.form_submit_button("Assess Readmission Risk", type="primary")

        if submitted:
            patient_data = {
                'age': age,
                'gender': gender,
                'admission_source_id': admission_source,
                'time_in_hospital': time_in_hospital,
                'num_medications': num_medications,
                'num_lab_procedures': num_lab_procedures,
                'A1Cresult': a1c_result,
                'diabetesMed': diabetes_med,
                'change': change
            }
            
            with st.spinner("Analyzing patient profile with AI..."):
                prediction = self.predictor.predict(patient_data)
                probabilities = self.predictor.predict_proba(patient_data)
            
            st.markdown("---")
            st.subheader("📋 Assessment Result")
            
            if prediction == '<30':
                st.error(f"**HIGH ALERT:** High risk of readmission in less than 30 days (Probability: {probabilities['<30']*100:.1f}%)")
            elif prediction == '>30':
                st.warning(f"**MODERATE ALERT:** Risk of readmission after 30 days (Probability: {probabilities['>30']*100:.1f}%)")
            else:
                st.success(f"**LOW RISK:** Patient with high probability of not being readmitted (Probability: {probabilities['NO']*100:.1f}%)")
            
            st.markdown("**Probability Details:**")
            st.progress(probabilities['NO'], text=f"No Readmission (NO): {probabilities['NO']*100:.1f}%")
            st.progress(probabilities['<30'], text=f"Less than 30 days (<30): {probabilities['<30']*100:.1f}%")
            st.progress(probabilities['>30'], text=f"More than 30 days (>30): {probabilities['>30']*100:.1f}%")

    def render_main_content(self):
        if self.page == "Major Stats (EDA)":
            st.title("Diabetics Readmission - Main Stats")
            st.markdown("---")
            
            if not self.eda_plots:
                st.warning("No EDA plots added yet.")
                return

            cols = st.columns(2)
            for i, plot in enumerate(self.eda_plots):
                with cols[i % 2]: 
                    plot.render()

        elif self.page == "Frequent Pattern for readmission (Mining)":
            st.title("Frequent Pattern for readmission / Common health conditions that lead to readmission")
            st.markdown("---")
            
            if not self.pattern_plots:
                st.warning("No Pattern plots added yet.")
                return

            for plot in self.pattern_plots:
                plot.render()
                
        elif self.page == "Predictor (ML Model)":
            self.render_prediction_page()

    def run(self):
        st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
        self.render_sidebar()
        self.render_main_content()