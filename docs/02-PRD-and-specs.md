# Especificação de Requisitos e Escopo do Produto (Gate 2 / PRD)

Documento de especificação funcional, hipóteses estatísticas e requisitos técnicos para o pipeline analítico e preditivo do projeto `improve_traffic`.

---

## 1. Visão Geral e Problema Central

**Declaração em uma frase**: Construir um pipeline modular em Python para diagnosticar os impactos do clima e da malha urbana no congestionamento de Nova York, estimando a duração de corridas de táxi com modelos de machine learning e publicando os resultados em um relatório técnico via Quarto no GitHub Pages.

### 1.1. Perfil dos Usuários e Decisores
- **Planejadores de Mobilidade e Tráfego Urbano**: Interessados na quantificação dos gargalos espaciais, impacto de eventos extremos de neve/chuva e recomendações para escoamento viário.
- **Operadores de Frota e Logística**: Focados na previsibilidade do tempo de viagem por faixa horária para alocação eficiente de veículos e precificação dinâmica.
- **Avaliadores Técnicos e Cientistas de Dados**: Focados no rigor estatístico, reprodutibilidade do pipeline, ausência de vazamento temporal e adequação das métricas de machine learning.

---

## 2. Consolidação Metodológica: Fases ASK e PREPARE

### 2.1. Questões Centrais de Negócio (Fase ASK)
1. Qual é o impacto marginal de chuvas e nevascas no tempo de deslocamento urbano em comparação com dias secos?
2. Em que proporção os horários de pico (manhã e tarde) amplificam a duração de viagens que cruzam Manhattan?
3. Viagens com origem ou destino nos aeroportos (JFK e LaGuardia) apresentam dinâmicas distintas das viagens intramunicipais?
4. É possível prever com precisão operacional a duração de uma corrida utilizando apenas coordenadas de partida/chegada, timestamp e indicadores climáticos do dia?

### 2.2. Hipóteses Estatísticas para Teste
- **H1 (Elasticidade Climática)**: Dias com registro de precipitação $> 5\text{ mm}$ ou acúmulo de neve no solo aumentam o tempo médio de viagem em mais de 15% e elevam a variância dos percursos.
- **H2 (Gargalo de Manhattan)**: Viagens que ocorrem integralmente dentro da malha de Manhattan sofrem degradação de velocidade média superior a 30% nos picos comerciais (07h-09h e 16h-19h) em relação ao horário entre-picos.
- **H3 (Dinâmica Aeroportuária)**: Trajetos envolvendo JFK e LaGuardia apresentam distâncias maiores e velocidades médias superiores, sendo menos vulneráveis a chuvas leves, porém altamente sensíveis a congestionamentos nas vias expressas de acesso (ex.: Van Wyck Expressway e Grand Central Parkway).

### 2.3. Definição da Variável Alvo e Escala de Modelagem
A variável alvo é `trip_duration` (segundos). Devido à forte assimetria à direita na distribuição de tempos de percurso, o treinamento de modelos de regressão será conduzido sobre a escala transformada:

$$z = \log(1 + y) = \log(1 + \text{trip\_duration})$$

Para avaliação e inferência, as previsões são revertidas pela exponencial ($\hat{y} = \exp(\hat{z}) - 1$).

---

## 3. Requisitos do Sistema

### 3.1. Requisitos Funcionais (RF)

| ID | Descrição | Fase Metodológica |
| :--- | :--- | :--- |
| **RF01** | Ingerir os arquivos brutos (`train.csv`, `test.csv`, `weather.csv`) validando os contratos de entrada do Gate 1. | Prepare |
| **RF02** | Converter unidades climáticas de `weather.csv` para o Sistema Internacional (°C para temperatura e mm para precipitação/neve). | Process |
| **RF03** | Mapear valores `"T"` (traço) nas colunas de chuva e neve para `0.0` e converter as colunas para numéricas. | Process |
| **RF04** | Filtrar ruído operacional no conjunto de treino, removendo viagens com `trip_duration < 60` segundos ou `trip_duration > 7200` segundos (2 horas). | Process |
| **RF05** | Calcular métricas de distância espacial: distância Haversine (linha reta) e distância Manhattan (em malha retangular urbana), além do ângulo de deslocamento (bearing). | Process |
| **RF06** | Extrair atributos de calendário e pico a partir de `pickup_datetime` (hora, dia da semana, mês, flag de fim de semana e categorias de horário de pico). | Process |
| **RF07** | Executar join temporal das viagens com a base climática usando a data de embarque (`pickup_date`). | Process |
| **RF08** | Implementar estratégia de validação cruzada com corte temporal, preservando a segunda metade de junho para teste de validação e evitando vazamento entre períodos. | Analyze |
| **RF09** | Treinar e comparar modelos preditivos: Baseline Linear (Ridge), Árvore de Decisão/Random Forest, LightGBM e XGBoost. | Analyze |
| **RF10** | Avaliar os modelos utilizando RMSLE (primária) e MAE em minutos (operacional). | Analyze |
| **RF11** | Gerar análise de interpretabilidade e importância de variáveis via SHAP values ou importância baseada em ganho. | Analyze / Share |
| **RF12** | Estruturar relatório executivo e técnico em Quarto (`.qmd`) com gráficos interpretáveis e narrativa orientada a decisões de tráfego. | Share |
| **RF13** | Configurar automação de compilação e deploy das páginas do relatório no GitHub Pages. | Act |

### 3.2. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Descrição |
| :--- | :--- | :--- |
| **RNF01** | Reprodutibilidade | Todas as etapas de amostragem, split e treinamento devem ter sementes pseudoaleatórias fixadas (`random_state=42`). |
| **RNF02** | Modularidade | O código do pipeline deve ser organizado em módulos Python autocontidos (`src/`), com separação clara entre ingestão, engenharia de atributos, modelagem e utilitários. |
| **RNF03** | Eficiência | Cálculos de distâncias espaciais e transformações de datas devem ser vetorizados com NumPy e Pandas, sem laços explícitos de iteração sobre linhas. |
| **RNF04** | Portabilidade | As dependências do projeto devem ser integralmente declaradas em arquivo padronizado (`environment.yml` / `pyproject.toml`). |
| **RNF05** | Clareza Narrativa | O relatório Quarto deve evitar jargões desnecessários, priorizando visualizações diretas de fácil leitura por gestores públicos e equipes operacionais. |

---

## 4. Métricas de Sucesso e Critérios de Aceite

1. **Desempenho de Machine Learning**:
   - **Métrica Primária**: RMSLE inferior a **0.40** no conjunto de validação temporal (benchmark alinhado aos modelos competitivos do Kaggle/DataCamp para o conjunto de táxis de NYC).
   - **Métrica Secundária**: MAE médio inferior a **4,5 minutos** em corridas urbanas típicas.
2. **Qualidade do Relatório**:
   - Documento Quarto compilado em HTML estático sem dependências quebradas de imagem ou scripts.
   - Presença obrigatória de seções dedicadas para cada uma das seis fases analíticas (*Ask, Prepare, Process, Analyze, Share, Act*).
3. **Publicação Contínua**:
   - Relatório acessível via URL pública no GitHub Pages.

---

## 5. Non-Goals (O Que Não Será Feito Nesta Versão)

Para garantir foco e entrega rápida sem dispersão técnica, os seguintes itens estão explicitamente fora do escopo:

- **Não haverá API REST em tempo real**: O projeto não disponibilizará endpoints de inferência (FastAPI, Flask ou Docker) para consumo em produção.
- **Não haverá aplicativo web ou dashboard em tempo real**: Não serão desenvolvidas interfaces com streaming de dados ou mapas ao vivo com websockets.
- **Não haverá consulta externa a APIs de roteamento**: Não serão consumidas chamadas a serviços pagos ou remotos (Google Maps Directions, Mapbox ou OSRM) durante a execução; as distâncias serão aproximadas matematicamente via Haversine e Manhattan.
- **Não haverá previsão de demanda por zona via grafos pesados**: O escopo foca na duração de viagens individuais (`trip_duration`) e fatores de tráfego, sem construir modelos complexos de previsão de frota agregada por grafos espaciais (GNNs).
- **Não haverá orquestração complexa de nuvem**: Não serão utilizados Airflow em cluster remoto, Kubeflow ou serviços gerenciados de nuvem paga nesta entrega. O pipeline rodará de forma local e modular.

---

## 6. Próximos Passos (Transição para o Gate 3)

Com a aprovação formal deste PRD, avançamos ao **Gate 3 (Arquitetura de Software, Domínio e Modelagem do Pipeline)**, onde serão definidos:
- Desenho dos módulos de código Python (`src/data`, `src/features`, `src/models`).
- Definição do fluxo das 4 fases finais (*Process, Analyze, Share, Act*).
- Esquema de diretórios físicos e contratos de interface entre etapas.
- Registros de decisão arquitetural (ADRs).
