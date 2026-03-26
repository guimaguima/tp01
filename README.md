# Previsão de Readmissão Hospitalar para Pacientes Diabéticos

## 🎯 Objetivos e Principais Features
O objetivo deste projeto é estudar a previsão de readmissão hospitalar em menos de 30 dias para pacientes diabéticos, com base em seus registros clínicos e histórico de tratamentos. Essa proposta vem com a intenção de observar quais comorbidades, intervenções e mudanças na medicação mais agravam o risco de retorno ao hospital, bem como estruturar a melhor modelagem para prever tal evento. Para isso, torna-se crucial analisar as relações complexas entre as variáveis, avaliar a incerteza e a variância dos estimadores, e construir um pipeline de limpeza utilizando Pandas e Scikit-learn que garanta um tratamento de dados matematicamente sólido ao longo deste mês. Os principais tipos de features envolvidas no processo são:

* Dados demográficos e de internação: como faixa etária, tempo de permanência no hospital e tipo de admissão inicial.

* Diagnósticos e histórico clínico: abrangendo os códigos CID (primários e secundários) da visita, além do número de visitas ambulatoriais e idas à emergência no ano anterior.

* Medicamentos e procedimentos: com foco na quantidade de exames laboratoriais realizados durante a internação e nas alterações de dosagem de medicamentos cruciais, especialmente a insulina.

### 💡 Principais Hipóteses
Nossas principais hipóteses se alinham com nosso interesse em modelar o retorno da hospitalização de portadores de diabetes, procurando observar os principais fatores que impactam nesse comportamento. A partir dessa premissa, desejamos observar diferentes comportamentos de subgrupos, verificar a correlação de atributos clínicos e construir modelos preditivos e explicativos para a tendência de retorno de internação.

### 📊 Dataset e Detalhamento das Features
🔗 [Diabetes 130 US hospitals for years 1999-2008](https://www.kaggle.com/datasets/brandao/diabetes/data) - 101766 linhas x 47 colunas

O conjunto de dados representa 10 anos (1999-2008) de atendimento clínico em 130 hospitais e redes integradas de saúde dos EUA. Inclui mais de 47 características que representam os desfechos dos pacientes e dos hospitais. As informações foram extraídas do banco de dados para atendimentos que atendiam aos seguintes critérios:

* É um atendimento de internação (admissão hospitalar).

* É um atendimento relacionado a diabetes, ou seja, um atendimento no qual qualquer tipo de diabetes foi registrado no sistema como diagnóstico.

* A duração da internação foi de no mínimo 1 dia e no máximo 14 dias.

* Foram realizados exames laboratoriais durante o atendimento.

* Foram administrados medicamentos durante o atendimento.

Os dados contêm atributos como número do paciente, raça, sexo, idade, tipo de admissão, tempo de internação, especialidade médica do médico responsável pela admissão, número de exames laboratoriais realizados, resultado do teste de HbA1c, diagnóstico, número de medicamentos, medicamentos para diabetes, número de consultas ambulatoriais, internações e visitas ao pronto-socorro no ano anterior à hospitalização, etc. (47 features).

Os dados estão baixados neste repositório em /data/diabetic_data.csv e o mapeamento das colunas em /data/IDS_mapping.csv. Dentro do .csv constam todas as 101766 observações de diferentes pacientes que seguem os critérios definidos acima.

# 👥 Membros da Equipe e Papéis
(Observação: Adicionar o 4º membro para cumprir o requisito da disciplina)

Gabriel Guimarães dos Santos Ricardo: [1] Analista de Dados / [2] Cientista de Dados (Aplicação e interpretação de modelos / criação de dashboard)

Enzo de Souza Braz: [1] Engenheiro de Dados / [2] Cientista de Dados (tratamento e limpeza dos dados / análise exploratória e clusterização)

Eduardo Birchal: Cientista de Dados (modelagem exploratória)

# 🛠️ Pilha de Tecnologias
As tecnologias previstas para o MVP são:

## 💻 Ambiente e Gestão

Máquinas Próprias: (Máquina menos potente: Intel i7, 16GB de RAM, Placa de vídeo MX110).

## 🐍 Linguagem

Python: 3.12+

## 📚 Bibliotecas (Versões atualizadas)

Pandas (3.0.1): Para a manipulação dos dados.

NumPy (2.4.3): Cálculo vetorial.

Scikit-Learn (1.8.0): Algoritmos de Machine Learning.

SciPy (1.17.1): Cálculo Numérico.

Statsmodels (0.14.6): Modelagem estatística avançada.

Streamlit (1.55.0): Criação rápida de dashboards web interativos.
