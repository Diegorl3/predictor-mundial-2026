# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Configuración de página
st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")
DATA_DIR = "data"

@st.cache_resource
def load_ai():
    try:
        with open(os.path.join(DATA_DIR, 'xgboost_model.pkl'), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'team_encoder.pkl'), 'rb') as f: encoder = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'current_elo.pkl'), 'rb') as f: current_elo = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'current_form.pkl'), 'rb') as f: current_form = pickle.load(f)
        with open(os.path.join(DATA_DIR, 'team_values.pkl'), 'rb') as f: team_values = pickle.load(f)
            
        model.set_params(device="cpu")
        return model, encoder, current_elo, current_form, team_values
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}")
        return None, None, None, None, None

model, encoder, current_elo, current_form, team_values = load_ai()
st.title("🏆 Predictor Mundial FIFA 2026")
st.markdown("### Análisis impulsado por Machine Learning (XGBoost + Poisson ELO)")
st.divider()

if model:
    equipos_disponibles = sorted(encoder.classes_)
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.subheader("🏠 Equipo Local")
        equipo_a = st.selectbox("Selecciona el primer equipo", equipos_disponibles, index=equipos_disponibles.index('France') if 'France' in equipos_disponibles else 0)

    with col3:
        st.subheader("✈️ Equipo Visitante")
        equipo_b = st.selectbox("Selecciona el segundo equipo", equipos_disponibles, index=equipos_disponibles.index('Senegal') if 'Senegal' in equipos_disponibles else 1)

    with col2:
        st.write("")
        st.write("")
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)

    st.write("---")
    
    if st.button("🚀 Ejecutar Predicción Analítica", use_container_width=True):
        if equipo_a == equipo_b:
            st.warning("Por favor, selecciona dos equipos diferentes.")
        else:
            with st.spinner('Procesando algoritmos predictivos...'):
                team_a_enc = encoder.transform([equipo_a])[0]
                team_b_enc = encoder.transform([equipo_b])[0]
                
                elo_a = current_elo.get(equipo_a, 1500)
                elo_b = current_elo.get(equipo_b, 1500)
                form_a = current_form.get(equipo_a, 0)
                form_b = current_form.get(equipo_b, 0)
                
                # Obtener valores de mercado (Fallback a 50M)
                val_a = team_values.get(equipo_a, 50)
                val_b = team_values.get(equipo_b, 50)
                
                X_pred = pd.DataFrame([[
                    team_a_enc, team_b_enc, 1, 0, elo_a, elo_b, (elo_a - elo_b), form_a, form_b, (val_a - val_b)
                ]], columns=model.feature_names_in_)

                probs = model.predict_proba(X_pred)[0]
                prob_derrota, prob_empate, prob_victoria = probs[0], probs[1], probs[2]

                # --- MOTOR POISSON ---
                # Traducir la probabilidad de la IA a Goles Esperados (xG) base 1.2
                xg_a = max(0.3, 1.2 + ((elo_a - elo_b) / 500) + (prob_victoria - prob_derrota))
                xg_b = max(0.3, 1.2 + ((elo_b - elo_a) / 500) + (prob_derrota - prob_victoria))

                # Crear matriz de marcadores (0 a 5 goles)
                max_goles = 6
                score_matrix = np.zeros((max_goles, max_goles))
                for i in range(max_goles):
                    for j in range(max_goles):
                        score_matrix[i][j] = poisson.pmf(i, xg_a) * poisson.pmf(j, xg_b)
                
                # Normalizar la matriz para que sume 100%
                score_matrix = (score_matrix / score_matrix.sum()) * 100
                most_likely_idx = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
                marcador_probable = f"{most_likely_idx[0]} - {most_likely_idx[1]}"

                # --- RENDERIZADO DE MÉTRICAS ---
                st.subheader("📊 Resultados de la Simulación")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(label=f"Victoria {equipo_a}", value=f"{prob_victoria*100:.2f}%", delta=f"xG: {xg_a:.2f}")
                m2.metric(label="Empate", value=f"{prob_empate*100:.2f}%", delta_color="off")
                m3.metric(label=f"Victoria {equipo_b}", value=f"{prob_derrota*100:.2f}%", delta=f"xG: {xg_b:.2f}")
                m4.metric(label="Marcador Exacto", value=marcador_probable, delta="Más Probable", delta_color="normal")

                st.divider()

                # --- ZONA DE GRÁFICOS VISUALES ---
                st.subheader("📈 Dashboard Analítico")
                g1, g2 = st.columns(2)

                with g1:
                    fig_probs, ax_probs = plt.subplots(figsize=(6, 4))
                    fig_probs.patch.set_facecolor('none')
                    ax_probs.set_facecolor('none')
                    
                    labels = [equipo_a, 'Empate', equipo_b]
                    valores = [prob_victoria*100, prob_empate*100, prob_derrota*100]
                    colores = ['#2ecc71', '#95a5a6', '#e74c3c']
                    
                    bars = ax_probs.barh(labels, valores, color=colores)
                    ax_probs.set_xlabel('Probabilidad (%)', color='white')
                    ax_probs.tick_params(colors='white')
                    for spine in ax_probs.spines.values(): spine.set_color('white')
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax_probs.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', color='white')
                    
                    st.pyplot(fig_probs)

                with g2:
                    labels_radar = ['Poder ELO', 'Forma Reciente', 'Ventaja IA']
                    angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
                    angles += angles[:1]

                    radar_a = [min(elo_a/200, 10), min(form_a/1.5, 10), prob_victoria*10]
                    radar_b = [min(elo_b/200, 10), min(form_b/1.5, 10), prob_derrota*10]
                    radar_a += radar_a[:1]
                    radar_b += radar_b[:1]

                    fig_radar, ax_radar = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
                    fig_radar.patch.set_facecolor('none')
                    ax_radar.set_facecolor('none')

                    ax_radar.plot(angles, radar_a, color='#3498db', linewidth=2, label=equipo_a)
                    ax_radar.fill(angles, radar_a, color='#3498db', alpha=0.25)
                    ax_radar.plot(angles, radar_b, color='#e74c3c', linewidth=2, label=equipo_b)
                    ax_radar.fill(angles, radar_b, color='#e74c3c', alpha=0.25)

                    ax_radar.set_xticks(angles[:-1])
                    ax_radar.set_xticklabels(labels_radar, color='white')
                    ax_radar.set_yticklabels([])
                    ax_radar.spines['polar'].set_color('white')
                    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1e1e1e', edgecolor='none', labelcolor='white')
                    
                    st.pyplot(fig_radar)

                # --- MAPA DE CALOR DE MARCADORES (POISSON) ---
                st.write("---")
                st.subheader(f"🔥 Probabilidad de Marcador Exacto (Distribución de Poisson)")
                
                fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                fig_hm.patch.set_facecolor('none')
                
                # Configurar Seaborn para tema oscuro
                sns.heatmap(score_matrix, annot=True, fmt=".1f", cmap="mako", 
                            cbar_kws={'label': 'Probabilidad (%)'}, ax=ax_hm, 
                            annot_kws={"color": "white", "weight": "bold"})
                
                ax_hm.set_xlabel(f"Goles de {equipo_b}", color='white', fontsize=12)
                ax_hm.set_ylabel(f"Goles de {equipo_a}", color='white', fontsize=12)
                ax_hm.tick_params(colors='white')
                
                # Cambiar el color del texto de la barra de color
                cbar = ax_hm.collections[0].colorbar
                cbar.ax.yaxis.label.set_color('white')
                cbar.ax.tick_params(colors='white')
                
                st.pyplot(fig_hm)