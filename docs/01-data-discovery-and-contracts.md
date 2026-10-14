# Diagnóstico de Dados e Contratos de Interface (Gate 1)

Documento técnico de auditoria das fontes brutas de dados do projeto `improve_traffic`. O objetivo deste documento é catalogar os esquemas, atestar a integridade das bases, identificar anomalias físicas e congelar os contratos de dados antes de qualquer etapa de processamento ou modelagem.

---

## 1. Contexto Inicial e Alinhamento do Problema (Fase ASK)

O projeto investiga os determinantes da dinâmica de trânsito em Nova York com base no histórico de viagens de táxi amarelo no primeiro semestre de 2016, enriquecido com séries meteorológicas diárias do Central Park.

### 1.1. Objetivos Centrais
- **Predição de Duração de Viagens**: Construir modelo supervisionado para estimar o tempo total de percurso (`trip_duration`) a partir de atributos espaciais, temporais e climáticos.
- **Identificação de Gargalos Urbanos**: Mapear padrões de congestionamento associados a horários de pico, eventos meteorológicos (neve e chuva intensa) e zonas de alta densidade como Manhattan e aeroportos (JFK, LaGuardia).
- **Suporte à Tomada de Decisão**: Formular recomendações operacionais e rotas alternativas para apoiar a gestão do tráfego urbano em cenários climáticos adversos.

### 1.2. Métrica Alvo de Avaliação
A avaliação primária da competição utiliza a raiz do erro quadrático médio logarítmico (RMSLE). A métrica penaliza desvios proporcionais relativos e lida bem com a assimetria da distribuição de tempo:

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2}$$

Como métrica operacional complementar, calcula-se o erro absoluto médio (MAE) em minutos para facilitar a interpretação gerencial dos desvios.

---

## 2. Catálogo de Dados e Esquemas Brutos (Fase PREPARE)

Os dados brutos residem no diretório `data/` divididos em três arquivos CSV. Todos os arquivos foram inspecionados diretamente via Python sem alterações em disco.

### 2.1. Arquivo `data/train.csv` (Treinamento)
- **Dimensões**: 60.554 linhas e 11 colunas.
- **Valores Nulos**: 0 registros ausentes em todas as colunas.
- **Período Coberto**: De 01/01/2016 00:08:07 a 30/06/2016 23:45:21 (182 dias, exatamente 6 meses completos).
- **Volume Mensal**: Cerca de 10.000 corridas por mês (Jan: 9.611; Fev: 9.897; Mar: 10.531; Abr: 10.616; Mai: 10.276; Jun: 9.623).

| Coluna | Tipo Lido | Tipo Destino | Descrição | Restrições / Valores Observados |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `str` | `string` | Identificador único da corrida | Chave primária (alfanumérico, ex: `id1234567`) |
| `vendor_id` | `int64` | `category` | Código do operador do táxi | Valores 1 (46,9%) ou 2 (53,1%) |
| `pickup_datetime` | `str` | `datetime64[ns]` | Timestamp de embarque do passageiro | `2016-01-01 00:08:07` a `2016-06-30 23:45:21` |
| `dropoff_datetime` | `str` | `datetime64[ns]` | Timestamp de desembarque do passageiro | `2016-01-01 00:19:25` a `2016-07-01 16:37:39` |
| `passenger_count` | `int64` | `int8` | Quantidade de passageiros | Varia de 1 a 6 (Mediana: 1; 70,6% viagens com 1 pessoa) |
| `pickup_longitude` | `float64` | `float64` | Longitude de embarque | Mín: -74.3936; Máx: -73.5181; Média: -73.9735 |
| `pickup_latitude` | `float64` | `float64` | Latitude de embarque | Mín: 40.4447; Máx: 41.3115; Média: 40.7520 |
| `dropoff_longitude` | `float64` | `float64` | Longitude de desembarque | Mín: -74.3936; Máx: -73.5181; Média: -73.9735 |
| `dropoff_latitude` | `float64` | `float64` | Latitude de desembarque | Mín: 40.4447; Máx: 41.3115; Média: 40.7520 |
| `store_and_fwd_flag` | `str` | `category` | Registro em memória do taxímetro | `N` (99,41%, 60.198) ou `Y` (0,59%, 356) |
| `trip_duration` | `int64` | `int32` | Duração da viagem em segundos (**alvo**) | Mín: 1 s; Máx: 86.387 s; Mediana: 661 s (11,0 min) |

### 2.2. Arquivo `data/test.csv` (Teste / Inferência)
- **Dimensões**: 47.999 linhas e 9 colunas.
- **Valores Nulos**: 0 registros ausentes em todas as colunas.
- **Período Coberto**: De 16/06/2016 12:47:45 a 30/06/2016 23:59:58 (apenas 15 dias).
- **Disparidade de Amostragem**: O teste não possui distribuição uniforme ao longo de 2016. Concentra-se inteiramente na segunda metade de junho, com média de 3.200 viagens por dia.
- **Sobreposição de IDs**: Zero IDs coincidentes entre treino e teste.

| Coluna | Tipo Lido | Tipo Destino | Descrição | Restrições / Valores Observados |
| :--- | :--- | :--- | :--- | :--- |
| `id` | `str` | `string` | Identificador único da corrida | Chave primária de submissão |
| `vendor_id` | `int64` | `category` | Código do operador do táxi | Valores 1 (47,0%) ou 2 (53,0%) |
| `pickup_datetime` | `str` | `datetime64[ns]` | Timestamp de embarque | `2016-06-16 12:47:45` a `2016-06-30 23:59:58` |
| `passenger_count` | `int64` | `int8` | Quantidade de passageiros | Varia de 0 a 9 (inclui anomalias: 1 registro com 0 e 1 com 9) |
| `pickup_longitude` | `float64` | `float64` | Longitude de embarque | Mín: -74.3936; Máx: -73.4836 |
| `pickup_latitude` | `float64` | `float64` | Latitude de embarque | Mín: 40.4485; Máx: 41.1065 |
| `dropoff_longitude` | `float64` | `float64` | Longitude de desembarque | Mín: -74.3264; Máx: -73.4836 |
| `dropoff_latitude` | `float64` | `float64` | Latitude de desembarque | Mín: 40.5263; Máx: 41.1065 |
| `store_and_fwd_flag` | `str` | `category` | Registro em memória do taxímetro | `N` (99,30%, 47.663) ou `Y` (0,70%, 336) |

### 2.3. Arquivo `data/weather.csv` (Medições Climáticas)
- **Dimensões**: 366 linhas e 7 colunas (cobre os 366 dias do ano bissexto de 2016 completo, de 01/01/2016 a 31/12/2016).
- **Valores Nulos Formais**: 0 nulos detectados pelo parser de texto.
- **Chave de Data**: Formato textual `D-M-YYYY` (ex: `1-1-2016`, `23-1-2016`), com dia antes do mês. Sem duplicatas.

| Coluna | Tipo Lido | Tipo Destino | Descrição Original | Unidade Real Comprovada | Presença de "T" |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `date` | `str` | `date` | Data da medição diária | Formato `D-M-YYYY` | 0 |
| `maximum temperature` | `int64` | `float32` | Temperatura máxima do dia | **Graus Fahrenheit (°F)** | 0 |
| `minimum temperature` | `int64` | `float32` | Temperatura mínima do dia | **Graus Fahrenheit (°F)** | 0 |
| `average temperature` | `float64` | `float32` | Temperatura média diária | **Graus Fahrenheit (°F)** | 0 |
| `precipitation` | `str` | `float32` | Precipitação diária total | **Polegadas (*inches*)** | 24 dias com `"T"` |
| `snow fall` | `str` | `float32` | Queda de neve acumulada no dia | **Polegadas (*inches*)** | 15 dias com `"T"` |
| `snow depth` | `str` | `float32` | Espessura de neve no solo | **Polegadas (*inches*)** | 7 dias com `"T"` |

---

## 3. Descobertas Críticas, Riscos e Mapeamento de Relacionamentos

### 3.1. Correção de Unidades Meteorológicas
A documentação prévia no `README.md` indicava que as temperaturas estavam em Celsius e a precipitação em milímetros. A inspeção minuciosa dos dados refutou essa premissa:
1. No dia 14/02/2016, a base registra mínima de -1 e máxima de 15. Esse evento coincide com o vórtice polar histórico registrado em Nova York, onde os termômetros bateram -1°F (-18,3°C) no Central Park.
2. No dia 23/01/2016, a base registra queda de neve de 27.3. Essa data marca a nevasca Jonas (segunda maior nevasca da história de Nova York), com exatamente 27,3 polegadas de neve acumulada.
3. Se 27.3 fossem milímetros, representaria meros 2,7 centímetros de neve, o que não geraria paralisia viária.

**Diretriz**: Os dados climáticos estão no padrão imperial norte-americano (°F e polegadas). No pipeline de processamento (`Process`), as variáveis serão mantidas puras ou convertidas explicitamente para Celsius e milímetros com documentação rastreável.

### 3.2. Tratamento da Flag "T" (Trace)
A letra `"T"` indica ocorrência meteorológica residual abaixo do limite mensurável pelo pluviômetro convencional (geralmente menor que 0,01 polegada para chuva e 0,1 polegada para neve).
- Na etapa `Process`, todo `"T"` deve ser mapeado numericamente para `0.0` (ou para constante residual `0.001` caso se deseje distinguir dias secos de dias com garoa leve).
- A coluna deve ser convertida para `float32` logo após a substituição.

### 3.3. Viabilidade do Join Temporal
- A base `weather.csv` possui 366 registros contínuos cobrindo todo o ano de 2016.
- Os 182 dias de embarque presentes em `train.csv` encontram correspondência de 100% no arquivo de clima.
- Os 15 dias de embarque de `test.csv` também possuem 100% de correspondência.
- **Chave de Junção**: Extrair a parte inteira da data do timestamp de embarque:
  $$\text{pickup\_date} = \text{cast}(\text{pickup\_datetime as date})$$
  Cruzamento com `weather.parsed_date`.

### 3.4. Risco Estrutural de Data Leakage e Diferença Treino/Teste
A comparação de esquemas entre treino e teste evidenciou dois pontos fundamentais:
1. **Ausência de `dropoff_datetime` no Teste**: A coluna `dropoff_datetime` existe exclusivamente no treino. Ela não pode ser utilizada como variável explicativa sob hipótese alguma, pois revela o tempo de viagem diretamente:
   $$\text{trip\_duration} = \text{dropoff\_datetime} - \text{pickup\_datetime}$$
   Seu uso causaria vazamento perfeito de dados.
2. **Distribuição Temporal Assimétrica**:
   - O treino se estende de janeiro a junho com densidade de 10.000 amostras/mês.
   - O teste reúne 47.999 amostras concentradas entre 16 e 30 de junho.
   - Em junho, o treino possui apenas 4.749 viagens na mesma janela quinzenal.
   - **Impacto na Modelagem**: Uma divisão aleatória de treino e validação (K-Fold aleatório simples) superestimará o desempenho em estações frias que não aparecem no conjunto de teste. A estratégia de validação cruzada do Gate 3 precisará levar em conta o corte temporal e a sazonalidade de junho.

### 3.5. Anomalias Físicas e Outliers Detectados
- **Durações Extremas de Corrida**:
  - 204 corridas duram menos de 30 segundos (77 duram menos de 10 segundos, com registros de 1 e 2 segundos). Decorrem de cancelamentos imediatos, testes de aparelho ou erros operacionais.
  - 77 corridas duram mais de 12 horas, atingindo até 86.387 segundos (23 horas e 59 minutos). Trata-se de taxímetros esquecidos ligados pelo motorista, encerrados pelo sistema central ao atingir o limite diário de 24 horas.
- **Corridas com Deslocamento Zero**:
  - 239 corridas no treino e 206 corridas no teste possuem coordenadas de embarque idênticas às de desembarque (distância calculada = 0 km).
  - No treino, a mediana dessas viagens é de 374 segundos (~6 minutos), alcançando até 1 hora. Representam clientes que pediram espera ou viagens de circuito fechado.
- **Passageiros no Teste**:
  - O conjunto de teste contém 1 corrida com 0 passageiros e 1 com 9 passageiros. No treino, os valores variam de 1 a 6. O pipeline de inferência precisa aceitar esses valores sem gerar erro de pré-processamento.
- **Limites Geográficos**:
  - A quase totalidade das coordenadas situa-se na área metropolitana de Nova York (-74.39 a -73.48 longitude e 40.44 a 41.31 latitude). Apenas 53 registros do treino e 45 do teste ultrapassam os limites estritos da cidade, conectando condados vizinhos de Nova Jersey, Long Island e Connecticut. Nenhuma coordenada foi gravada como (0, 0).

---

## 4. Contratos de Dados (Data Contracts)

Para proteger a integridade do pipeline antes de iniciar a fase `Process`, estabelecem-se as seguintes regras de validação que os dados de entrada devem satisfazer:

### 4.1. Contrato de Entrada: `train.csv`
1. **Completude de Colunas**: Devem estar presentes as 11 colunas originais (`id`, `vendor_id`, `pickup_datetime`, `dropoff_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`, `trip_duration`).
2. **Não-Nulidade**: Nenhuma coluna admite valores nulos (`null_count == 0`).
3. **Unicidade de Identificador**: A coluna `id` deve ser única para cada linha.
4. **Alvo Válido**: `trip_duration` deve ser inteiro estritamente positivo ($> 0$).
5. **Datas Coerentes**: `dropoff_datetime >= pickup_datetime`.
6. **Coordenadas Reais**: Longitudes entre -75.0 e -72.0; Latitudes entre 40.0 e 42.0.

### 4.2. Contrato de Entrada: `test.csv`
1. **Completude de Colunas**: Devem estar presentes as 9 colunas (`id`, `vendor_id`, `pickup_datetime`, `passenger_count`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `store_and_fwd_flag`).
2. **Proibição de Leakage**: As colunas `dropoff_datetime` e `trip_duration` não devem existir.
3. **Não-Nulidade**: Nenhuma coluna admite valores nulos.
4. **Tolerância a Categorias Exógenas**: `passenger_count` fora do intervalo $[1, 6]$ deve ser tratado pelo transformador sem lançar exceção.

### 4.3. Contrato de Entrada: `weather.csv`
1. **Estrutura**: 7 colunas esperadas (`date`, `maximum temperature`, `minimum temperature`, `average temperature`, `precipitation`, `snow fall`, `snow depth`).
2. **Volumetria do Calendário**: Exatamente 366 registros cobrindo o ano de 2016 completo.
3. **Tipagem Flexível de Chuva e Neve**: Admitir caracteres `'T'` nas colunas pluviométricas e de neve, com conversão obrigatória para `float32`.
4. **Consistência de Temperatura**: `maximum temperature >= minimum temperature`.

---

## 5. Próximos Passos (Transição para o Gate 2)

Com a auditoria concluída e os contratos formalizados, a base de dados está homologada para as fases seguintes. A próxima etapa contempla o **Gate 2 (PRD e Especificação Analítica)**, consolidando:
- Perguntas de negócio e hipóteses para a fase *Process* e *Analyze*.
- Delimitação de requisitos funcionais, métricas de sucesso e lista de *non-goals*.
- Formulação da estrutura do relatório no Quarto e estratégia de deploy no GitHub Pages.
