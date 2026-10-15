# Changelog

Todas as alterações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [0.2.0] - 2026-10-14

### Adicionado
- Suíte de testes de caracterização (*Golden Master*) em `tests/characterization/test_pinning_pipeline.py`, travando schemas, ausência de nulos e invariantes numéricos.
- Governança de cache no manifesto Quarto (`report/_quarto.yml`) com `execute: freeze: auto` e versionamento de `report/_freeze/`.
- Registro de decisão arquitetural em `docs/adr/0004-characterization-testing-and-freeze-governance.md`.

### Modificado
- Vetorização de regras condicionais em `src/evaluate.py` utilizando `np.select` para classificação de impacto climático e rotas de aeroportos.
- Sincronização automática de figuras entre `reports/figures/` e `report/figures/` no script de avaliação.
- Centralização de constantes de domínio geográfico (JFK, LGA), faixas de pico e limiares climáticos no topo de `src/features.py`.
- Reorganização de importações em `src/train.py` conforme convenções da PEP 8.
- Atualização estrutural do `README.md` com resultados de benchmark dos 4 modelos, métricas do estimador LightGBM e guia de execução do `Makefile`.

### Corrigido
- Ajuste editorial em `report/index.qmd` eliminando adjetivação redundante nas seções de diagnóstico.

## [0.1.0] - 2026-10-14

### Adicionado
- Pipeline modular em `src/` (`utils.py`, `data_prep.py`, `features.py`, `train.py`, `evaluate.py`).
- Persistência intermediária em Apache Parquet em `data/processed/`.
- Suíte de testes unitários com 13 casos em `tests/`.
- Treinamento e comparação de 4 modelos supervisionados (Ridge, Random Forest, LightGBM e XGBoost) com validação temporal.
- Relatório técnico Quarto estruturado em 6 fases de Data Analytics (`report/index.qmd`).
- Workflow de automação do GitHub Actions em `.github/workflows/deploy-report.yml`.
