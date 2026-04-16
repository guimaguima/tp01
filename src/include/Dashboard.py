import streamlit as st
import pandas as pd
from include.DiabeticDataLoader import DiabeticDataLoader
from include.Plot import Plot

class Dashboard:
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df = self._load_and_prep_data()
        self.eda_plots = []  
        self.pattern_plots = []

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
            ["Major Stats (EDA)", "Frequent Pattern for readmission (Mineração)"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.header("INFO")
        st.sidebar.info(f"Total Patients Analyzed: {len(self.df)}")

    def render_main_content(self):
        # Renderiza o conteúdo dependendo da página selecionada
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

        elif self.page == "Frequent Pattern for readmission (Mineração)":
            st.title("Frequent Pattern for readmission / Common health conditions that lead to readmission")
            st.markdown("---")
            
            if not self.pattern_plots:
                st.warning("No Pattern plots added yet.")
                return

            for plot in self.pattern_plots:
                plot.render()

    def run(self):
        st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
        self.render_sidebar()
        self.render_main_content()