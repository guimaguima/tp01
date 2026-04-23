import streamlit as st
import altair as alt
import pandas as pd
from abc import ABC, abstractmethod
from include.PatternMiner import PatternMiner
  

class Plot(ABC):
    """
    Abstract base class for all dashboard plots.
    Enforces a consistent interface for rendering visualizations.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def generate_chart(self) -> alt.Chart:
        """s
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

class AssociationRulesPlot(Plot):
    
    def __init__(self,data = None,csv_path: str="../data/processed/rules_association.csv"):  
        self.csv_path = csv_path
        self.pattern_miner = PatternMiner.from_csv(csv_path)
        self.data = data
    
    def generate_chart(self) -> alt.Chart:
        pass
    
    def render(self):
        # controles organizados em 3 colunas para ficar elegante
        col1, col2, col3 = st.columns(3)
        with col1:
            targets = ["readmitted_NO", "readmitted_<30", "readmitted_>30"]
            selected_target = st.selectbox("Readmission (Consequent):", targets)
        with col2:
            min_sup = st.slider("Minimum Probability", 0.01, 0.20, 0.05, 0.01)
        with col3:
            min_conf = st.slider("Minimum Conditional Probability", 0.01, 0.50, 0.05, 0.01)
        
        csv_path = "../data/processed/rules_association.csv"
        

            
        if not self.pattern_miner:
            st.error(f"File not found at: {csv_path}. Please check the path.")
            return

        df_rules = self.pattern_miner.filter_patterns(column=selected_target, min_support=min_sup, min_confidence=min_conf, top_n=5)
        
        if df_rules.empty:
            st.warning(f"No strong rules found for {selected_target}. Try reducing the support or confidence.")
            return

        melted_rules = df_rules.melt(
            id_vars=['antecedents_str'], 
            value_vars=['confidence', 'lift', 'odds_ratio'],
            var_name='Metric', 
            value_name='Value'
        )
        
        base_chart = alt.Chart(melted_rules).mark_bar().encode(
            x=alt.X('Value:Q', title=None),
            y=alt.Y('antecedents_str:N', title='', sort='-x', axis=alt.Axis(labelLimit=400)),
            color=alt.Color('Metric:N', legend=None, scale=alt.Scale(scheme='set2')),
            tooltip=['antecedents_str', 'Metric', 'Value']
        ).properties(
            height=150
        )

        faceted_chart = base_chart.facet(
            row=alt.Row('Metric:N', title=None, header=alt.Header(labelFontSize=14, labelFontWeight='bold'))
        ).resolve_scale(
            x='independent' 
        )

        st.altair_chart(faceted_chart, use_container_width=True)