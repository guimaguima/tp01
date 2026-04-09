import streamlit as st
import altair as alt
import pandas as pd
from abc import ABC, abstractmethod
from include.DiabeticDataLoader import DiabeticDataLoader
from include.Plot import Plot

class Dashboard:
    
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.df = self._load_and_prep_data()
        self.plots = []

    @st.cache_data 
    def _load_and_prep_data(_self) -> pd.DataFrame:
        # Load raw CSV
        df_raw = pd.read_csv(_self.raw_data_path)
        
        # Initialize your existing data loader
        loader = DiabeticDataLoader(df_raw)
        
        df_clean = loader.get_clean_data() 
        df_no_outliers = loader.get_no_outliers_data() 
        return df_no_outliers

    def add_plot(self, plot_class: type[Plot]):
        # Instantiate the plot with the loaded data
        plot_instance = plot_class(self.df)
        self.plots.append(plot_instance)

    def render_sidebar(self):
        st.sidebar.header("Dashboard Controls")
        st.sidebar.info(f"Total Patients Analyzed: {len(self.df)}")

    def render_main_content(self):
        st.title("Diabetics Readmission - Readmission main stats")
        st.markdown("---")
        
        if not self.plots:
            st.warning("No plots added to the dashboard yet.")
            return

        # Render plots in a 2-column layout as an example
        cols = st.columns(2)
        for i, plot in enumerate(self.plots):
            with cols[i % 2]: # Alternate between column 0 and 1
                plot.render()

    def run(self):
        st.set_page_config(page_title="Diabetes EDA Dashboard", layout="wide")
        self.render_sidebar()
        self.render_main_content()