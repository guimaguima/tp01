
import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from include.Dashboard import Dashboard
from include.Plot import ReadmissionDistributionPlot,TimeInHospitalVsReadmissionPlot


if __name__ == "__main__":
    # Define the path to your data file
    DATA_PATH = "../data/raw/diabetic_data.csv" # Update this to your actual path
    
    # Initialize Dashboard
    app = Dashboard(DATA_PATH)
    
    app.add_plot(ReadmissionDistributionPlot)
    app.add_plot(TimeInHospitalVsReadmissionPlot)
    
    app.run()