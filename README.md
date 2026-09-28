# 🥋 Judo ML & Elo/Glicko-2 Analytics

Plataforma de **Machine Learning**, **Rating Elo Ponderado**, **Glicko-2** e **Simulador Interativo** para análise de dados históricos de Judô baseada em 184.949 lutas e 21.715 atletas ranqueados.

---

## 📂 Dataset & Fontes de Dados

Os dados utilizados neste projeto são originários do repositório oficial e do dataset no Kaggle mantidos por [@caiosaboia](https://github.com/caiosaboia):

* 🐙 **GitHub:** [caiosaboia/judo_data](https://github.com/caiosaboia/judo_data)
* 📊 **Kaggle:** [Judo Matchs IJF Dataset](https://www.kaggle.com/datasets/caiosaboia/judo-matchs-ijf-dataset)

### Como obter o dataset:
Para manter o repositório leve, o arquivo `judo_atle_compt.csv` (~42.8 MB) não é versionado diretamente no Git. Para executar o projeto:
1. Baixe o arquivo `judo_atle_compt.csv` no [Kaggle](https://www.kaggle.com/datasets/caiosaboia/judo-matchs-ijf-dataset) ou pelo [GitHub judo_data](https://github.com/caiosaboia/judo_data).
2. Salve o arquivo na pasta `data/` do projeto:
   ```text
   data/judo_atle_compt.csv
   ```

---

## 🚀 Como Executar com Docker (Recomendado)

### 1. Iniciar a Aplicação Web (Streamlit Dashboard)
Para construir a imagem e subir o painel web interativo na porta `8501`:
```bash
docker compose up --build
```
Acesse no seu navegador: **[http://localhost:8501](http://localhost:8501)**

### 2. Rodar em Segundo Plano (Modo Detached)
```bash
docker compose up -d
```
Para parar os containers:
```bash
docker compose down
```

### 3. Executar o Pipeline Principal de ML via Docker
Para rodar a pipeline de treinamento, cálculo de ratings e geração de gráficos sem abrir o dashboard:
```bash
docker compose run --rm pipeline
```

### 4. Executar os Testes Unitários via Docker
```bash
docker compose run --rm test
```

---

## 💻 Como Executar Localmente com Python & `uv`

### Pré-requisitos
* Python `>=3.13`
* [`uv`](https://docs.astral.sh/uv/) instalado

### Instalação e Execução
```bash
# Executar a pipeline principal de treinamento e gráficos
uv run python main.py

# Iniciar o dashboard web interativo
uv run streamlit run app.py

# Executar a suíte de testes unitários (20 testes)
uv run pytest

# Verificação de qualidade de código (Ruff)
uv run ruff check .
```

---

## 📊 Arquitetura e Modelos de Machine Learning

* **Sistemas de Pontuação:**
  * **Elo Dinâmico:** Pesos por relevância de competição ($W \in \{1, 2, 3\}$).
  * **Glicko-2:** Modelagem bayesiana com desvio de incerteza ($RD$) e volatilidade ($\sigma$).
  * **Elo por Margem:** Multiplicadores por Ippon ($1.5\times$), Wazari ($1.2\times$) e Shido/Decisão ($1.0\times$).
* **Modelos Preditivos:**
  * **LightGBM Classifier:** 67.47% de Acurácia | 0.7289 ROC-AUC.
  * **Random Forest Classifier:** 66.93% de Acurácia | 0.7217 ROC-AUC.
  * **Victory Method Multi-Class:** Previsão de desfecho (Ippon, Wazari ou Penalidade/Shido).

Para mais detalhes matemáticos e estatísticos, consulte o [`GEMINI.md`](GEMINI.md).
