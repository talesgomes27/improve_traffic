# ADR 0002: Estratégia de Validação Temporal com Corte em Junho

- **Status**: Aprovado
- **Data**: 2026-10-14
- **Decisores**: Tales (Desenvolvedor / Cientista de Dados)

---

## Contexto e Problema

O conjunto de dados de treino (`data/train.csv`) contém 60.554 viagens distribuídas de forma homogênea entre 01/01/2016 e 30/06/2016 (~10.000 amostras por mês). O conjunto de teste (`data/test.csv`), por sua vez, contém 47.999 amostras concentradas estritamente na segunda quinzena de junho (16/06/2016 a 30/06/2016).

Se utilizarmos validação cruzada aleatória simples (K-Fold padrão com embaralhamento uniforme):
1. O modelo aprenderá com dados do futuro para prever o passado.
2. O score de validação refletirá uma média ponderada dos 6 meses (incluindo inverno rigoroso e primavera amena), enquanto o teste real ocorre no início do verão nova-iorquino.
3. Haverá risco de vazamento temporal e sobreajuste (*overfitting*) que não se sustentará nos dados de teste.

---

## Decisão

Adotaremos uma estratégia de validação cruzada orientada por corte temporal:

1. **Conjunto de Validação Principal (Holdout Temporal)**:
   - Treino de Modelagem: Viagens ocorridas de 01/01/2016 a 15/06/2016 (55.805 registros).
   - Validação de Modelagem: Viagens ocorridas de 16/06/2016 a 30/06/2016 (4.749 registros).
2. **Espelhamento do Problema de Inferência**: Essa partição replica com precisão a tarefa real de prever a segunda quinzena de junho a partir de dados históricos anteriores.
3. **Métrica de Corte**: O hiperparâmetro ótimo e a seleção do melhor modelo (LightGBM vs XGBoost vs Ridge) serão definidos pelo RMSLE obtido no subconjunto de validação temporal de 16 a 30 de junho.
4. **Retreinamento Final**: Após congelar os hiperparâmetros, o modelo vencedor poderá ser treinado em todo o conjunto de treino histórico (janeiro a junho) antes de gerar as previsões finais para `data/test.csv`.

---

## Consequências

### Positivas
- Medição fidedigna da capacidade de generalização do modelo em dados futuros.
- Proteção total contra vazamento de variáveis dependentes de tempo e sazonalidade climática.

### Negativas
- O conjunto de validação fica limitado a 4.749 amostras, o que exige atenção com possíveis variações estatísticas concentradas em dias pontuais de chuva no fim de junho.
