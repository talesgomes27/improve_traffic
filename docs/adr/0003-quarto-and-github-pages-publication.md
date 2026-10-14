# ADR 0003: Publicação Contínua de Relatório Técnico via Quarto e GitHub Pages

- **Status**: Aprovado
- **Data**: 2026-10-14
- **Decisores**: Tales (Desenvolvedor / Cientista de Dados)

---

## Contexto e Problema

O projeto precisa entregar um relatório técnico e executivo completo, cobrindo as seis fases de Data Analytics (*Ask, Prepare, Process, Analyze, Share, Act*).

A apresentação do relatório precisa atender a três requisitos:
1. Publicação acessível por qualquer navegador sem necessidade de instalar Python ou rodar notebooks localmente.
2. Formato estático seguro, com suporte nativo a código embutido, tabelas formatadas e gráficos estatísticos.
3. Esteira automatizada de build e publicação integrada ao repositório no GitHub.

---

## Decisão

Adotaremos o **Quarto** como gerador do relatório estático e o **GitHub Pages** como plataforma de hospedagem:

1. **Estrutura do Relatório Quarto**:
   - Um arquivo de configuração central `_quarto.yml` na raiz ou no diretório de documentação/relatório.
   - O documento principal do relatório será redigido em `report/index.qmd`, estruturado estritamente nos blocos *Ask, Prepare, Process, Analyze, Share, Act*.
   - A compilação gerará arquivos HTML/CSS estáticos na pasta de saída `_site/` ou `docs/site/`.
2. **Integração com o Pipeline de Dados**:
   - Os gráficos pesados e resumos estatísticos gerados por `src/evaluate.py` serão exportados para `reports/figures/` e `reports/tables/`.
   - O documento `.qmd` consumirá essas figuras e tabelas ou executará blocos leves de visualização direta sem refazer treinamentos caros de modelos.
3. **Automação via GitHub Actions**:
   - Criaremos um workflow (`.github/workflows/deploy-report.yml`) disparado em pushes na branch `main`.
   - A action instalará o Quarto, configurará as dependências de renderização e fará o deploy automático para a branch `gh-pages` ou pasta `docs/`.

---

## Consequências

### Positivas
- Visual limpo, profissional e responsivo, ideal para demonstrar o trabalho técnico.
- Independência de servidores dinâmicos pagos; hospedagem estática gratuita no GitHub Pages.
- O versionamento do código-fonte do relatório permanece acoplado ao código do modelo.

### Negativas
- Requer instalação local do binário do Quarto para compilação durante o desenvolvimento.
- Exige atenção aos caminhos relativos de imagens para evitar quebras de links na URL do GitHub Pages.
