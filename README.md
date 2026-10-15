# improve-traffic: Previsão de Tráfego e Duração de Viagens em Nova York

Pipeline de ponta a ponta para análise e previsão do tempo de deslocamento (`trip_duration`) de viagens de táxi em Nova York, combinando dados de mobilidade da NYC Taxi and Limousine Commission (TLC) e medições meteorológicas diárias de 2016.

O projeto resolve o desafio *"Can You Find a Better Way to Manage Traffic?"* da plataforma DataCamp com engenharia de atributos vetorizada, validação temporal estrita e modelos de *Gradient Boosting*.

## Resultados e Benchmark

A validação foi conduzida em partição temporal *holdout* (treino com dados até 15/06/2016 e validação na segunda quinzena de junho de 2016 com 4.711 amostras), espelhando o período do conjunto de teste sem vazamento de dados.

O **LightGBM** obteve o melhor desempenho entre os estimadores avaliados, superando a meta de projeto ($\text{RMSLE} \le 0.40$):

| Modelo | RMSLE | RMSE (min) | MAE (min) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.3397** | **5.45** | **3.25** | **0.7934** |
| XGBoost | 0.3425 | 5.43 | 3.27 | 0.7946 |
| Random Forest | 0.3807 | 6.00 | 3.65 | 0.7497 |
| Ridge Baseline | 0.6074 | 11.66 | 5.83 | 0.0560 |

As distâncias espaciais calculadas (Haversine e Manhattan) concentram 72,9% do ganho preditivo total. Dias com acúmulo de neve reduzem a velocidade média em 24,3%, enquanto horários de pico comercial da tarde (16h às 19h) registram a menor fluidez média em dias úteis (12,8 km/h).

## Estrutura do Repositório

```text
improve_traffic/
├── data/
│   ├── train.csv              # 60.554 viagens brutas de treino
│   ├── test.csv               # 47.999 viagens brutas de teste
│   ├── weather.csv            # 366 observações meteorológicas diárias de 2016
│   └── processed/             # Parquets higienizados e submission.csv
├── models/
│   ├── best_model.joblib      # Modelo LightGBM campeão serializado
│   └── metrics_summary.json   # Métricas JSON de todos os estimadores
├── reports/
│   ├── figures/               # Gráficos diagnósticos em 300 DPI
│   └── tables/                # Tabelas comparativas de métricas
├── report/
│   ├── _quarto.yml            # Configuração do site Quarto (freeze: auto)
│   ├── index.qmd              # Relatório estruturado em 6 fases analíticas
│   └── _site/                 # Build estático pronto para GitHub Pages
├── src/
│   ├── utils.py               # Funções matemáticas puras e distâncias vetorizadas
│   ├── data_prep.py           # Ingestão, conversão de unidades e saneamento
│   ├── features.py            # Atributos espaciais, temporais e climáticos
│   ├── train.py               # Split temporal, treino dos 4 modelos e métricas
│   └── evaluate.py            # Diagnósticos visuais e inferência de teste
├── tests/
│   ├── test_utils.py          # Testes unitários do núcleo matemático
│   ├── test_data_prep.py      # Testes de contrato de dados e clima
│   ├── test_features.py       # Testes das regras de atributos
│   ├── test_models.py         # Testes do particionamento temporal
│   └── characterization/      # Testes de caracterização (Golden Master)
├── environment.yml            # Especificação do ambiente Conda/Mamba
├── Makefile                   # Automação do ciclo de dados e compilação
└── README.md                  # Documentação do projeto
```

## Configuração do Ambiente

O projeto utiliza Miniforge3 com Mamba ou Conda no canal `conda-forge`:

```bash
# Criação do ambiente isolado
mamba env create -f environment.yml

# Ativação do ambiente
conda activate improve-traffic

# Registro do kernel para JupyterLab (opcional)
python -m ipykernel install --user --name improve-traffic --display-name "Python (improve-traffic)"
```

## Execução do Pipeline

O ciclo completo de reprodução de dados, treino e compilação é orquestrado via `Makefile`:

```bash
# Executa todo o pipeline: prepare, features, train, evaluate e report
make all

# Ou etapas individuais:
make prepare     # Saneamento e join temporal (gera clean_train e clean_test)
make features    # Extração de 28 features espaciais e climáticas
make train       # Treinamento dos 4 modelos com corte em 16/06/2016
make evaluate    # Geração de gráficos, tabelas e data/processed/submission.csv
make report      # Compilação do relatório HTML Quarto em report/_site/
make test        # Execução de toda a suíte de testes unitários e de caracterização
```

## Relatório Técnico e Storytelling

O relatório completo está redigido em Quarto ([report/index.qmd](file:///home/tales/Documentos/github/improve_traffic/report/index.qmd)) e estruturado nas 6 fases de Data Analytics (*Ask, Prepare, Process, Analyze, Share, Act*):
- Fundamentação matemática da escolha do RMSLE frente ao RMSE tradicional.
- Análise de impacto de tempestades e frio extremo na velocidade média.
- Diagnóstico de resíduos e dispersão Real vs. Previsto.
- Recomendações operacionais para despacho de frotas e janelas de segurança para aeroportos (JFK e LaGuardia).

O site estático compilado localiza-se em `report/_site/index.html` e possui publicação automatizada no GitHub Pages via GitHub Actions.
