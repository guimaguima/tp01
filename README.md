# Previsão de Possibilidade de doença cardíaca

## Objetivos e Principais Features

O objetivo deste projeto é estudar a previsão de readmissão hospitalar em menos de 30 dias para pacientes diabéticos, com base em seus registros clínicos e histórico de tratamentos. Essa proposta vem com a intenção de observar quais comorbidades, intervenções e mudanças na medicação mais agravam o risco de retorno ao hospital, bem como estruturar a melhor modelagem para prever tal evento. Para isso, torna-se crucial analisar as relações complexas entre as variáveis, avaliar a incerteza e a variância dos estimadores, e construir um pipeline de limpeza utilizando Pandas e Scikit-learn que garanta um tratamento de dados matematicamente sólido ao longo deste mês. As principais features envolvidas no processo são:

*  Dados demográficos e de internação, como faixa etária, tempo de permanência no hospital e tipo de admissão inicial.

*  Diagnósticos e histórico clínico, abrangendo os códigos CID (primários e secundários) da visita, além do número de visitas ambulatoriais e idas à emergência no ano anterior.

*  Medicamentos e procedimentos, com foco na quantidade de exames laboratoriais realizados durante a internação e nas alterações de dosagem de medicamentos cruciais, especialmente a insulina.

### Principais Hipóteses

Nossas principais hipóteses se alinham com nosso interesse em modelar o retorno da hospitalização de portadores de diabetes, procurando observar os principais fatores que impactam nesse comportamento. A partir dessa premissa desejamos observar diferentes comportamentos de subgrupos, verificar correlação de atributos clínicos e contruir modelos preditivos, assim como explicativos, para a tendência de retorno da internação em hospitais.

## Dataset

[Diabetes 130 US hospitals for years 1999-2008]([https://www.kaggle.com/datasets/brandao/diabetes/data](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)) - 3 MB

O conjunto de dados representa 10 anos (1999-2008) de atendimento clínico em 130 hospitais e redes integradas de saúde dos EUA. Inclui mais de 50 características que representam os desfechos dos pacientes e dos hospitais. As informações foram extraídas do banco de dados para atendimentos que atendiam aos seguintes critérios:

*   É um atendimento de internação (admissão hospitalar).
*   É um atendimento relacionado a diabetes, ou seja, um atendimento no qual qualquer tipo de diabetes foi registrado no sistema como diagnóstico.
*   A duração da internação foi de no mínimo 1 dia e no máximo 14 dias.
*   Foram realizados exames laboratoriais durante o atendimento.
*   Foram administrados medicamentos durante o atendimento.

Os dados contêm atributos como número do paciente, raça, sexo, idade, tipo de admissão, tempo de internação, especialidade médica do médico responsável pela admissão, número de exames laboratoriais realizados, resultado do teste de HbA1c, diagnóstico, número de medicamentos, medicamentos para diabetes, número de consultas ambulatoriais, internações e visitas ao pronto-socorro no ano anterior à hospitalização, etc. (50 features)

Os dados estão baixados neste repositório em /data/diabetic_data.csv e o mapeamento das colunas em /data/IDS_mapping.csv.

## Membros da Equipe e Papéis

*   Gabriel Guimarães dos Santos Ricardo : Analista de Dados / Cientista de Dados (aplicação e interpretação de modelos / criação dashboard)
*   Enzo de Souza Braz: Engenheiro de Dados / Cientista de Dados (tratamento e limpeza dos dados / analise exploratória e clusterização)
*   Eduardo Birchal: Cientista de Dados (modelagem exploratória)

## Pilha de Tecnologias

Usaremos como nossa linguagem de programação principal Python. As bibliotecas adjacentes a este trabalho são:
*   Pandas
*   Numpy
*   Scikit-Learn
*   Scipy
*   Streamlit (Dashboard)
