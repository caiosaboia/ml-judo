import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Insert project root into sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.advanced_models import (
    parse_victory_method_target,
    train_lightgbm_model,
    train_victory_method_model,
)
from src.config import BASE_K, CSV_PATH, ELO_SCALE, INITIAL_ELO
from src.data_processing import load_and_clean_data
from src.elo import compute_elo_ratings
from src.elo_advanced import compute_margin_weighted_elo
from src.features import preprocess_and_engineer_features
from src.glicko2 import compute_glicko2_ratings
from src.model import train_and_evaluate_model

# Page config
st.set_page_config(
    page_title="Judo Analytics & ML Simulator",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E293B; margin-bottom: 0rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 1rem; border: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# Data loading & Caching
@st.cache_data(show_spinner="Carregando e processando 184.949 lutas...")
def load_all_pipeline_data():
    df_clean = load_and_clean_data(CSV_PATH)
    df_elo, final_elo = compute_elo_ratings(df_clean, initial_elo=INITIAL_ELO, base_k=BASE_K, scale=ELO_SCALE)
    _, final_glicko = compute_glicko2_ratings(df_clean)
    _, final_margin = compute_margin_weighted_elo(df_clean)
    
    # Merge Ratings into a consolidated df
    df_clean['blue_pre_elo'] = df_elo['blue_pre_elo']
    df_clean['white_pre_elo'] = df_elo['white_pre_elo']
    
    X, y = preprocess_and_engineer_features(df_clean)
    
    # Train Models
    rf_res = train_and_evaluate_model(X, y, split_ratio=0.8)
    lgbm_res = train_lightgbm_model(X, y, split_ratio=0.8)
    
    # Victory Method
    y_method = parse_victory_method_target(df_clean)
    method_res = train_victory_method_model(X, y_method, split_ratio=0.8)
    
    # Build Athlete Directory
    athlete_info = {}
    for row in df_clean.itertuples():
        b_name = f"{row.blue_first_name} {row.blue_last_name}"
        w_name = f"{row.white_first_name} {row.white_last_name}"
        
        b_id = row.athlete_blue_id
        w_id = row.athlete_white_id
        
        if b_id not in athlete_info:
            athlete_info[b_id] = {
                'id': b_id, 'name': b_name, 'gender': row.blue_gender,
                'weight_cat': row.weight_category, 'country': row.blue_country,
                'height': row.blue_height, 'weight': row.blue_weight,
                'elo': final_elo.get(b_id, 1500.0),
                'glicko': final_glicko[b_id].rating if b_id in final_glicko else 1500.0,
                'glicko_rd': final_glicko[b_id].rd if b_id in final_glicko else 350.0,
                'margin_elo': final_margin.get(b_id, 1500.0)
            }
            
        if w_id not in athlete_info:
            athlete_info[w_id] = {
                'id': w_id, 'name': w_name, 'gender': row.white_gender,
                'weight_cat': row.weight_category, 'country': row.white_country,
                'height': row.white_height, 'weight': row.white_weight,
                'elo': final_elo.get(w_id, 1500.0),
                'glicko': final_glicko[w_id].rating if w_id in final_glicko else 1500.0,
                'glicko_rd': final_glicko[w_id].rd if w_id in final_glicko else 350.0,
                'margin_elo': final_margin.get(w_id, 1500.0)
            }

    return df_clean, X, y, rf_res, lgbm_res, method_res, athlete_info

# Initialize Pipeline Data
df_clean, X, y, rf_res, lgbm_res, method_res, athlete_info = load_all_pipeline_data()

# App Title
st.markdown("<div class='main-header'>🥋 Judo Analytics & ML Simulator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Plataforma Interativa de Machine Learning, Sistema Elo & Glicko-2 para Análise de Judô</div>", unsafe_allow_html=True)

# Top Key Metrics Bar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total de Lutas", f"{len(df_clean):,}")
c2.metric("Atletas Ranqueados", f"{len(athlete_info):,}")
c3.metric("Baseline (Maior Elo)", f"{rf_res['baseline']['accuracy']*100:.1f}%")
c4.metric("Acurácia Random Forest", f"{rf_res['metrics']['accuracy']*100:.1f}%")
c5.metric("Acurácia LightGBM", f"{lgbm_res['metrics']['accuracy']*100:.1f}%")

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🥊 Simulador de Luta (Head-to-Head)",
    "🏆 Leaderboards & Rankings",
    "📈 Benchmarks de ML",
    "💡 Explorador de Insights"
])

# ---------------------------------------------------------
# TAB 1: MATCH SIMULATOR
# ---------------------------------------------------------
with tab1:
    st.subheader("Simulador de Combate em Tempo Real")
    st.write("Selecione dois atletas cadastrados no banco de dados ou configure os perfis manualmente para prever as probabilidades de vitória e a forma de término da luta.")
    
    col_mode, _ = st.columns([1, 2])
    sim_mode = col_mode.radio("Modo de Seleção:", ["Atletas do Banco de Dados", "Configuração Personalizada"], horizontal=True)
    
    athlete_list = sorted(athlete_info.values(), key=lambda x: x['name'])
    athlete_names = [f"{a['name']} ({a['country']} | Elo: {a['elo']:.0f})" for a in athlete_list]
    name_to_id = {f"{a['name']} ({a['country']} | Elo: {a['elo']:.0f})": a['id'] for a in athlete_list}
    
    col_b, col_w = st.columns(2)
    
    if sim_mode == "Atletas do Banco de Dados":
        with col_b:
            st.markdown("### 🔵 Atleta Blue")
            sel_b = st.selectbox("Selecione o Atleta Blue:", athlete_names, index=0, key="sel_b")
            a_b = athlete_info[name_to_id[sel_b]]
            st.info(f"**Nome:** {a_b['name']}\n**País:** {a_b['country']} | **Gênero:** {a_b['gender']}\n**Elo:** {a_b['elo']:.1f} | **Glicko-2:** {a_b['glicko']:.1f}\n**Altura:** {a_b['height']} cm | **Peso:** {a_b['weight']} kg")
            
        with col_w:
            st.markdown("### ⚪ Atleta White")
            sel_w = st.selectbox("Selecione o Atleta White:", athlete_names, index=min(1, len(athlete_names)-1), key="sel_w")
            a_w = athlete_info[name_to_id[sel_w]]
            st.info(f"**Nome:** {a_w['name']}\n**País:** {a_w['country']} | **Gênero:** {a_w['gender']}\n**Elo:** {a_w['elo']:.1f} | **Glicko-2:** {a_w['glicko']:.1f}\n**Altura:** {a_w['height']} cm | **Peso:** {a_w['weight']} kg")

        # Feature vector construction
        b_elo, w_elo = a_b['elo'], a_w['elo']
        b_h, w_h = float(a_b['height']) if a_b['height'] > 0 else 175.0, float(a_w['height']) if a_w['height'] > 0 else 175.0
        b_w, w_w = float(a_b['weight']) if a_b['weight'] > 0 else 73.0, float(a_w['weight']) if a_w['weight'] > 0 else 73.0
        weight_cat = a_b['weight_cat'] if pd.notna(a_b['weight_cat']) else 73.0
        is_male = 1 if a_b['gender'] == 'M' else 0
        b_country = a_b['country']
        w_country = a_w['country']
        
    else:
        with col_b:
            st.markdown("### 🔵 Atleta Blue (Custom)")
            b_elo = st.number_input("Rating Elo (Blue):", 1000, 2500, 1600, key="cust_b_elo")
            b_h = st.number_input("Altura (cm):", 140, 220, 175, key="cust_b_h")
            b_w = st.number_input("Peso (kg):", 40, 150, 73, key="cust_b_w")
            b_country = st.selectbox("País (Blue):", ["JAPAN", "BRAZIL", "FRANCE", "GEORGIA", "OTHER"], key="cust_b_c")
            
        with col_w:
            st.markdown("### ⚪ Atleta White (Custom)")
            w_elo = st.number_input("Rating Elo (White):", 1000, 2500, 1600, key="cust_w_elo")
            w_h = st.number_input("Altura (cm):", 140, 220, 175, key="cust_w_h")
            w_w = st.number_input("Peso (kg):", 40, 150, 73, key="cust_w_w")
            w_country = st.selectbox("País (White):", ["JAPAN", "BRAZIL", "FRANCE", "GEORGIA", "OTHER"], key="cust_w_c")
            
        weight_cat = b_w
        is_male = 1

    # Predict button
    if st.button("🚀 Simular Combate", type="primary", use_container_width=True):
        # Build Match Vector
        match_data = {
            'blue_pre_elo': b_elo, 'white_pre_elo': w_elo, 'elo_diff': b_elo - w_elo,
            'blue_height': b_h, 'white_height': w_h, 'height_diff': b_h - w_h,
            'blue_weight': b_w, 'white_weight': w_w, 'weight_diff': b_w - w_w,
            'weight_category': weight_cat, 'is_male': is_male
        }
        
        # Set dummy flags
        for col in X.columns:
            if col.startswith('blue_country_'):
                country_code = col.replace('blue_country_', '')
                match_data[col] = 1 if b_country == country_code else 0
            elif col.startswith('white_country_'):
                country_code = col.replace('white_country_', '')
                match_data[col] = 1 if w_country == country_code else 0
                
        df_single = pd.DataFrame([match_data]).reindex(columns=X.columns, fill_value=0)
        
        # Model predictions
        rf_prob = lgbm_res['model'].predict_proba(df_single)[0]
        prob_blue = rf_prob[1] * 100
        prob_white = rf_prob[0] * 100
        
        # Victory method
        method_probs = method_res['model'].predict_proba(df_single)[0]
        method_classes = method_res['classes']
        top_method_idx = np.argmax(method_probs)
        top_method = method_classes[top_method_idx].upper()
        
        st.markdown("### 📊 Resultado da Simulação ML")
        
        r1, r2 = st.columns(2)
        r1.metric("Probabilidade de Vitória (🔵 Blue)", f"{prob_blue:.1f}%")
        r2.metric("Probabilidade de Vitória (⚪ White)", f"{prob_white:.1f}%")
        
        st.progress(prob_blue / 100.0)
        
        st.success(f"**Método de Término mais Provável:** {top_method} (Confiança: {method_probs[top_method_idx]*100:.1f}%)")

# ---------------------------------------------------------
# TAB 2: LEADERBOARDS
# ---------------------------------------------------------
with tab2:
    st.subheader("Leaderboards de Atletas")
    st.write("Tabela interativa com os rankings de Elo tradicional, Glicko-2 e Elo Ponderado por Margem.")
    
    f_gen = st.multiselect("Filtrar por Gênero:", ["M", "F"], default=["M", "F"])
    
    df_ath = pd.DataFrame(list(athlete_info.values()))
    df_ath_filtered = df_ath[df_ath['gender'].isin(f_gen)].copy()
    
    search_query = st.text_input("🔍 Buscar Atleta por Nome:")
    if search_query:
        df_ath_filtered = df_ath_filtered[df_ath_filtered['name'].str.contains(search_query, case=False)]
        
    rank_type = st.radio("Ordenar por:", ["Elo Tradicional", "Glicko-2 Rating", "Elo Margem de Vitória"], horizontal=True)
    
    if rank_type == "Elo Tradicional":
        df_ath_filtered = df_ath_filtered.sort_values(by='elo', ascending=False)
    elif rank_type == "Glicko-2 Rating":
        df_ath_filtered = df_ath_filtered.sort_values(by='glicko', ascending=False)
    else:
        df_ath_filtered = df_ath_filtered.sort_values(by='margin_elo', ascending=False)
        
    st.dataframe(
        df_ath_filtered[['name', 'country', 'gender', 'weight_cat', 'elo', 'glicko', 'glicko_rd', 'margin_elo']].rename(
            columns={
                'name': 'Nome', 'country': 'País', 'gender': 'Gênero', 'weight_cat': 'Categoria Peso',
                'elo': 'Elo Base', 'glicko': 'Glicko-2', 'glicko_rd': 'Incerteza (RD)', 'margin_elo': 'Elo Margem'
            }
        ),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------
# TAB 3: MODEL BENCHMARKS
# ---------------------------------------------------------
with tab3:
    st.subheader("Comparativo de Modelos de Machine Learning")
    st.write("Métricas detalhadas comparando o modelo Baseline (Elo Puro), Random Forest Classifier e LightGBM Classifier.")
    
    b_df = pd.DataFrame({
        'Modelo': ['Baseline (Maior Elo Vence)', 'Random Forest Classifier', 'LightGBM Classifier'],
        'Acurácia': [rf_res['baseline']['accuracy'], rf_res['metrics']['accuracy'], lgbm_res['metrics']['accuracy']],
        'ROC-AUC': [rf_res['baseline']['roc_auc'], rf_res['metrics']['roc_auc'], lgbm_res['metrics']['roc_auc']],
        'F1-Score': ['-', f"{rf_res['metrics']['f1_score']:.4f}", f"{lgbm_res['metrics']['f1_score']:.4f}"]
    })
    
    st.table(b_df)
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("### Top 10 Feature Importances (Random Forest)")
        rf_feats = pd.DataFrame(rf_res['feature_ranking'], columns=['Feature', 'Importance'])
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rf_feats, x='Importance', y='Feature', palette='Blues_r', ax=ax_rf)
        ax_rf.set_title("Importância de Variáveis - RF")
        st.pyplot(fig_rf)
        
    with col_feat2:
        st.markdown("### Top 10 Feature Importances (LightGBM)")
        lgbm_feats = pd.DataFrame(lgbm_res['feature_ranking'], columns=['Feature', 'Importance'])
        fig_lg, ax_lg = plt.subplots(figsize=(6, 4))
        sns.barplot(data=lgbm_feats, x='Importance', y='Feature', palette='Oranges_r', ax=ax_lg)
        ax_lg.set_title("Importância de Variáveis - LightGBM")
        st.pyplot(fig_lg)

# ---------------------------------------------------------
# TAB 4: INSIGHTS EXPLORER
# ---------------------------------------------------------
with tab4:
    st.subheader("Explorador de Insights Estatísticos do Judô")
    
    i1, i2 = st.columns(2)
    
    with i1:
        st.markdown("#### 🥋 Viés da Cor do Judogi (Blue vs. White)")
        st.write("Atletas que usam judogi **Branco vencem 59%** das lutas contra **41% do Azul**. Isso ocorre porque o primeiro judoca chamado (Branco) é tipicamente o cabeça de chave no sistema IJF.")
        fig_c, ax_c = plt.subplots(figsize=(5, 3))
        ax_c.bar(['White (Branco)', 'Blue (Azul)'], [59.0, 41.0], color=['#E2E8F0', '#1E3A8A'])
        ax_c.set_ylabel("Taxa de Vitória (%)")
        st.pyplot(fig_c)
        
    with i2:
        st.markdown("#### 🇯🇵 Fator Dominância do Japão")
        st.write("Atletas japoneses vencem **79.1%** de seus confrontos contra atletas estrangeiros. Mesmo em lutas com ratings Elo equivalentes, o judoca do Japão vence **74.2%** das vezes.")
        fig_j, ax_j = plt.subplots(figsize=(5, 3))
        ax_j.bar(['Japão vs Outros', 'Outros Confrontos'], [79.1, 50.0], color=['#DC2626', '#94A3B8'])
        ax_j.set_ylabel("Taxa de Vitória (%)")
        st.pyplot(fig_j)
