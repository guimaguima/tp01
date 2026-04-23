import os
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from include.Dashboard import Dashboard
from include.Plot import AssociationRulesPlot, ReadmissionDistributionPlot,TimeInHospitalVsReadmissionPlot


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'diabetic_data.csv')
    
    app = Dashboard(DATA_PATH)
    
    # Adiciona na página 1 (EDA)
    app.add_eda_plot(ReadmissionDistributionPlot)
    app.add_eda_plot(TimeInHospitalVsReadmissionPlot)
    
    # Adiciona na página 2 (Mineração)
    app.add_pattern_plot(AssociationRulesPlot)
    
    app.run()