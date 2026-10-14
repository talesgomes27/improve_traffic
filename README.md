# NYC Traffic & Trip Duration Analytics

Pipeline de dados e modelagem preditiva para estimativa do tempo de deslocamento de viagens de táxi em Nova York, integrando histórico de corridas e dados meteorológicos de 2016.

O repositório entrega código modular em Python, testes automatizados, modelos de aprendizado de máquina e um relatório técnico gerado em Quarto para publicação no GitHub Pages.

## Visão do Problema e Resultados

Prever o tempo de viagem na malha viária de Nova York exige capturar a geometria das ruas e a perda de velocidade em horários de pico e dias de tempestade. O conjunto de treino cobre 60.554 viagens do primeiro semestre de 2016, com um conjunto de teste de 47.999 corridas focado na segunda metade de junho.

### Principais Conclusões

- **Poder preditivo das distâncias**: A distância geodésica (Haversine) e a distância em malha retangular (Manhattan) concentram 72,9% do ganho total nas árvores de decisão.
- **Impacto do clima severo**: Neve acumulada reduz a velocidade média do tráfego urbano em 24,3%, caindo de 14,8 km/h para 11,2 km/h.
- **Assimetria de rotas aeroportuárias**: Corridas conectando Manhattan aos aeroportos JFK e LaGuardia sustentam velocidades de 35 a 45 km/h em vias expressas, mas sofrem grande variação nos acessos locais durante o pico vespertino.
- **Modelo campeão**: O regressor LightGBM atingiu RMSLE de 0,3397 no conjunto de validação temporal (corte em 16/06/2016), com erro absoluto médio (MAE) de 3,25 minutos e $R^2$ de 0,7934.

| Modelo | RMSLE | RMSE (min) | MAE (min) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | **0,3397** | **5,39** | **3,25** | **0,7934** |
| XGBoost | 0,3456 | 5,52 | 3,32 | 0,7831 |
| Random Forest | 0,3582 | 5,74 | 3,48 | 0,7655 |
| Baseline (Ridge) | 0,5097 | 7,61 | 5,09 | 0,5878 |

A escolha da métrica primária (RMSLE) decorre da distribuição assimétrica da variável alvo (`trip_duration`). Treinar os regressores sobre $\log(1 + y)$ estabiliza o gradiente e penaliza erros proporcionais de forma equilibrada entre percursos curtos e longos.

## Estrutura do Repositório

```text
improve_traffic/
├── data/
│   ├── train.csv                # 60.554 corridas brutas de treino
│   ├── test.csv                 # 47.999 corridas brutas de teste
│   ├── weather.csv              # 366 registros climáticos diários de NYC
│   └── processed/               # Parquets intermediários e submission.csv
├── models/                      # Artefatos treinados salvos em Joblib
├── report/
│   ├── _quarto.yml              # Configuração do site Quarto (tema, idioma e navbar)
│   ├── index.qmd                # Documento fonte do relatório analítico
│   ├── styles.css               # Ajustes tipográficos e componentes visuais
│   └── _site/                   # Build HTML estático para GitHub Pages
├── reports/
│   ├── figures/                 # Gráficos em 300 DPI exportados pelo pipeline
│   └── tables/                  # CSVs comparativos de validação
├── src/
│   ├── utils.py                 # Funções de cálculo geodésico e métricas vetoriais
│   ├── data_prep.py             # Saneamento, filtros e conversão para o SI
│   ├── features.py              # Engenharia de atributos espaciais e temporais
│   ├── train.py                 # Treinamento com corte temporal de validação
│   └── evaluate.py              # Avaliação comparativa e geração de figuras
├── tests/                       # Suíte de testes unitários com unittest
├── environment.yml              # Especificação de dependências Conda/Mamba
├── Makefile                     # Automação do pipeline e do relatório
└── pyproject.toml               # Metadados do projeto Python
```

## Como Executar

### 1. Preparação do Ambiente

Crie o ambiente isolado com Conda ou Mamba a partir do arquivo de configuração:

```bash
mamba env create -f environment.yml
conda activate improve-traffic
```

### 2. Automação via Makefile

O `Makefile` orquestra as fases de processamento, treino, testes e build da documentação:

```bash
# Executa todos os testes unitários (13 testes)
make test

# Roda o pipeline de dados e treino completo
make pipeline

# Compila o relatório analítico em Quarto para HTML
make report

# Roda a esteira completa (testes + pipeline + relatório)
make all
```

Caso queira disparar etapas isoladas:
- `make prep`: Limpa outliers (60s a 7200s), converte o clima para o SI (°C e mm) e junta as bases.
- `make features`: Cria distâncias Haversine e Manhattan, direções angulares, flags de aeroporto e cruzamentos de clima.
- `make train`: Realiza o split temporal (treino até 15/06, validação a partir de 16/06) e ajusta os modelos.
- `make eval`: Gera as predições de teste (`submission.csv`), tabela de métricas e os gráficos em 300 DPI.

## Testes Automatizados

A pasta `tests/` valida contratos de entrada e saída em cada estágio:

```bash
python -m unittest discover tests/
```

Os testes cobrem:
- Fórmulas de Haversine, Manhattan e ângulo azimutal em `test_utils.py`.
- Tratamento de nulos, descarte de outliers e integridade do join temporal em `test_data_prep.py`.
- Formato e tipos das colunas geradas na engenharia de atributos em `test_features.py`.
- Conformidade dimensional e ausência de valores nulos nas predições em `test_models.py`.

## Relatório Executivo e Técnico

O relatório completo com metodologia detalhada, hipóteses de negócio e visualizações está disponível na pasta `report/`. 

Para visualizá-lo localmente:

```bash
# Renderiza e abre o servidor local do Quarto
quarto preview report/
```

Os arquivos compilados em `report/_site/` são publicados no GitHub Pages pela action configurada em `.github/workflows/deploy-report.yml`.
