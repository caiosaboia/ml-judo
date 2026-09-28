# Base de Conhecimento - Machine Learning, Elo & Glicko-2 para Judô

Este documento é a base de conhecimento (knowledge base) oficial do projeto **ml-judo**, registrando as decisões arquiteturais, sistemas de pontuação (Elo, Glicko-2 e Elo por Margem), modelagem preditiva com Random Forest e LightGBM, e instruções da aplicação interativa.

---

## 1. O Dataset de Judô
O projeto analisa o arquivo `data/judo_atle_compt.csv` com **184.949 lutas limpas** e **21.715 atletas ranqueados**.
O dataset inclui:
* Identificadores de atleta e cor do judogi (Blue vs. White).
* Nome e tipo de competição.
* Pontuações e punições (Ippon, Wazari, Shido, Hansoku-make).
* Características físicas (altura, peso, gênero, país).
* Histórico cronológico de confrontos.

---

## 2. Sistemas de Pontuação Implementados

### 2.1. Sistema Elo Tradicional (com Pesos por Competição)
* **Relevância de Competições ($W$):**
  * **Peso 3:** Jogos Olímpicos, Campeonatos Mundiais, World Masters.
  * **Peso 2:** Grand Slam, Grand Prix.
  * **Peso 1:** Campeonatos Continentais, Copas Continentais e Nacionais, Opens, Games.
* **Fórmula da Expectativa ($E_{blue}$):**
  $$E_{blue} = \frac{1}{1 + 10^{\frac{R_{white} - R_{blue}}{400}}}$$
* **Fator K Dinâmico:** $K = K_{base} \times W = 32 \times W$.

### 2.2. Sistema Glicko-2 (Com Incerteza $RD$ e Volatilidade $\sigma$)
Desenvolvido por Mark Glickman, o **Glicko-2** converte os ratings para a escala $\mu$ e $\phi$:
$$\mu = \frac{r - 1500}{173.7178}, \quad \phi = \frac{RD}{173.7178}$$
O sistema calcula a incerteza do atleta ($RD$) e reduz essa incerteza a cada combate realizado, permitindo ranquear atletas novatos vs. veteranos com maior precisão estatística.

### 2.3. Elo Ponderado por Margem de Vitória
Ajusta o fator $K$ de acordo com a forma como a luta foi encerrada:
* **Ippon:** Multiplicador $1.5\times$
* **Wazari:** Multiplicador $1.2\times$
* **Decisão / Punimentos (Shido):** Multiplicador $1.0\times$

---

## 3. Resultados dos Modelos de Machine Learning

Dividindo o dataset cronologicamente (80% treino / 20% teste):

| Modelo | Acurácia (Accuracy) | ROC-AUC | F1-Score |
| :--- | :---: | :---: | :---: |
| **Baseline (Maior Elo Vence)** | 63.99% | 0.6905 | N/A |
| **Random Forest Classifier** | 66.93% | 0.7217 | 0.5217 |
| **LightGBM Classifier** | **67.04%** | **0.7235** | **0.5231** |

### Predição de Método de Término de Luta (Multi-Class)
Treinamos um classificador multi-classe para prever a forma de encerramento da luta (**Ippon**, **Wazari**, ou **Shido/Decisão**), alcançando **>65% de acurácia multi-classe**.

---

## 4. Aplicação Web Interativa (`app.py` - Streamlit)

A plataforma conta com uma aplicação web interativa em **Streamlit** organizada em 4 abas:
1. **🥊 Simulador de Luta (Head-to-Head):** Escolha 2 atletas reais ou configure dados personalizados (Elo, altura, peso, país) para simular probabilidades de vitória e forma de encerramento.
2. **🏆 Leaderboards & Rankings:** Tabelas ordenáveis por Elo Base, Glicko-2 e Elo Margem, com filtros por gênero, peso e país.
3. **📈 Benchmarks de ML:** Comparativos de curva ROC, matriz de confusão e feature importances entre Random Forest e LightGBM.
4. **💡 Explorador de Insights:** Gráficos interativos abordando o viés do Judogi (Branco 59% vs Azul 41%), dominância japonesa e física da altura.

---

## 5. Como Executar o Projeto

### Executar a Pipeline Principal
```bash
uv run python main.py
```

### Iniciar o Dashboard Web Interativo (Streamlit)
```bash
uv run streamlit run app.py
```

### Executar a Suíte de Testes (Pytest)
```bash
uv run pytest
```

### Checagem de Código (Ruff)
```bash
uv run ruff check .
```

---

## 6. Execução via Docker (Containerização)

### Iniciar o Dashboard Web via Docker Compose
```bash
docker compose up --build
```
Acesse no navegador: `http://localhost:8501`

### Executar Pipeline via Docker
```bash
docker compose run --rm pipeline
```

### Executar Testes via Docker
```bash
docker compose run --rm test
```

