import pandas as pd
import numpy as np
import os

class PatternMiner:
    def __init__(self, rules_df: pd.DataFrame = None):
        self.all_rules = rules_df

    @classmethod
    def from_csv(cls, path: str):
        """Cria a classe carregando dados de um CSV."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            return cls(rules_df=df)
        return None

    def filter_patterns(self, column: str, min_support: float, min_confidence: float, top_n: int = 5) -> pd.DataFrame:
        if self.all_rules is None or self.all_rules.empty:
            return pd.DataFrame()

        # Filtro rápido no Pandas
        rules = self.all_rules[
            (self.all_rules['support'] >= min_support) & 
            (self.all_rules['confidence'] >= min_confidence) &
            (self.all_rules['consequents'].str.contains(column)) &
            (self.all_rules['lift'] > 1)
        ].copy()

        # Cálculo do Odds Ratio (baseado nas colunas do CSV)
        p11 = rules['support']
        p10 = rules['antecedent support'] - rules['support']
        p01 = rules['consequent support'] - rules['support']
        p00 = 1 - rules['antecedent support'] - rules['consequent support'] + rules['support']
        
        denominador = (p10 * p01).replace(0, np.nan)
        rules['odds_ratio'] = (p11 * p00) / denominador
        
        rules = rules.rename(columns={'antecedents': 'antecedents_str', 'consequents': 'consequent'})
        return rules[['antecedents_str', 'consequent', 'confidence', 'lift', 'odds_ratio']].sort_values(by='confidence', ascending=False).head(top_n)