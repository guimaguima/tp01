import streamlit as st
import altair as alt
import pandas as pd
from abc import ABC, abstractmethod

class Plot(ABC):
    """
    Abstract base class for all dashboard plots.
    Enforces a consistent interface for rendering visualizations.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def generate_chart(self) -> alt.Chart:
        """
        Builds and returns the Altair Chart object.
        Must be implemented by subclasses.
        """
        pass

    def render(self):
        """
        Renders the generated chart into the Streamlit app.
        """
        chart = self.generate_chart()
        st.altair_chart(chart, use_container_width=True)
        
        
class ReadmissionDistributionPlot(Plot):
    def generate_chart(self) -> alt.Chart:
        chart = alt.Chart(self.data).mark_bar().encode(
            x=alt.X('readmitted:N', title='Readmission Status', sort='-y'),
            y=alt.Y('count():Q', title='Number of Patients'),
            color=alt.Color('readmitted:N', legend=None),
            tooltip=['readmitted', 'count()']
        ).properties(
            title="Distribution of Patient Readmissions",
            height=400
        )
        return chart
    
class TimeInHospitalVsReadmissionPlot(Plot):
    def generate_chart(self) -> alt.Chart:
        chart = alt.Chart(self.data).mark_boxplot(extent='min-max').encode(
            x=alt.X('readmitted:N', title='Readmission Status'),
            y=alt.Y('time_in_hospital:Q', title='Days in Hospital'),
            color=alt.Color('readmitted:N', legend=None)
        ).properties(
            title="Time in Hospital by Readmission Status",
            height=400
        )
        return chart