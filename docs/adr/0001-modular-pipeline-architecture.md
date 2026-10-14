# ADR 0001: Arquitetura Modular do Pipeline em Scripts Python Desacoplados

- **Status**: Aprovado
- **Data**: 2026-10-14
- **Decisores**: Tales (Desenvolvedor / Cientista de Dados)

---

## Contexto e Problema

O projeto começou com um caderno Jupyter exploratório (`notebook.ipynb`). Embora notebooks facilitem a visualização rápida e a prototipação inicial, pipelines de dados baseados exclusivamente em cadernos únicos sofrem com problemas crônicos de manutenibilidade:
1. Estado global mutável e execução de células fora de ordem, dificultando a reprodutibilidade.
2. Dificuldade de reaproveitar funções lógicas em testes automatizados ou em relatórios dinâmicos.
3. Risco de acoplamento entre regras de saneamento de dados e rotinas de renderização visual.

Precisamos definir como o pipeline de dados, engenharia de atributos e modelagem será organizado para alimentar tanto experimentos locais quanto o relatório final no Quarto.

---

## Decisão

Adotaremos uma arquitetura modular baseada em scripts Python autocontidos dentro do diretório `src/`, complementados por um módulo utilitário de funções puras (`src/utils.py`):

1. **Separação em quatro estágios de execução**:
   - `src/data_prep.py`: Ingestão, conversão de unidades e junção temporal com o clima.
   - `src/features.py`: Engenharia de atributos espaciais, de calendário e interação climática.
   - `src/train.py`: Validação temporal e treinamento dos modelos de machine learning.
   - `src/evaluate.py`: Avaliação de métricas (RMSLE, MAE), geração de previsões e gráficos diagnósticos.
2. **Interface via Linha de Comando (CLI)**: Cada script poderá ser invocado individualmente no terminal ou em cadeia por um orquestrador leve.
3. **Persistência Intermediária em Apache Parquet**: Cada etapa salvará seus resultados em arquivos `.parquet` compactados com Snappy em `data/processed/`, garantindo preservação estrita de tipos e leitura rápida.
4. **Consumo no Quarto**: O documento do Quarto (`index.qmd`) consumirá os artefatos processados, tabelas e figuras geradas pelo pipeline, focando exclusivamente na narrativa técnica e storytelling.

---

## Consequências

### Positivas
- Código testável com funções puras e isolamento de dependências.
- Reprodutibilidade garantida por execução linear sem dependência de estado de notebook.
- O relatório Quarto compila de forma muito mais rápida, pois não precisa reprocessar distâncias espaciais ou retreinar modelos pesados a cada renderização.

### Negativas
- Exige disciplina para manter argumentos de CLI e contratos de entrada/saída documentados.
- Cria dependência de armazenamento local para arquivos intermediários em `data/processed/`.
