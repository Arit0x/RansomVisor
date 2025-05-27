import streamlit as st
from utils import load_css, carga_datos_victimas_por_ano
from sidebar import apply_filters, select_sector
import eda
from pathlib import Path
import base64
from utils import run_update_flow, show_update_summary

def home_app():

    # ─── CARGA DE ESTILOS ─────────────────────────────────────────────────────────
    load_css()

    # ─── BANNER ANIMADO CON IMAGEN ────────────────────────────────────────────────
    image_path = Path(__file__).parent.parent / "assets" / "portada.png"
    if image_path.exists():
        encoded_string = base64.b64encode(image_path.read_bytes()).decode()
        st.markdown(
            f"""
            <style>
            /* 1. Tipografía */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            /* 2. Animación */
            @keyframes fadeInUp {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to   {{ opacity: 1; transform: translateY(0); }}
            }}

            /* 3. Contenedor principal */
            .banner-container {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(30, 31, 45, 0.8);
                backdrop-filter: blur(6px);
                border-radius: 1rem;
                padding: 2rem;
                margin: 0 auto 2rem;
                max-width: 1200px;
                gap: 2rem;
                box-shadow: 0 12px 30px rgba(0,0,0,0.4);
                animation: fadeInUp 0.8s ease-out both;
            }}
            /* 4. Imagen */
            .banner-image {{
                flex: 1;
                min-width: 280px;
                max-width: 45%;
                transition: transform 0.3s ease;
            }}
            .banner-image img {{
                display: block;
                width: 100%;
                border-radius: 0.75rem;
                box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            }}
            .banner-image:hover {{
                transform: scale(1.02);
            }}
            /* 5. Texto */
            .banner-text {{
                flex: 1.2;
                font-family: 'Inter', sans-serif;
                color: #ECEFF1;
            }}
            .banner-text h1 {{
                font-size: 3rem;
                font-weight: 700;
                line-height: 1.2;
                margin: 0 0 1rem;
                word-break: break-word;
            }}
            .banner-text h1 .accent {{
                color: #00C0AD;
            }}
            .banner-text p {{
                font-size: 1rem;
                font-weight: 400;
                line-height: 1.6;
                margin: 0;
                max-width: 580px;
                word-break: break-word;
            }}
            /* 6. Responsive */
            @media (max-width: 768px) {{
                .banner-container {{
                    flex-direction: column;
                    text-align: center;
                }}
                .banner-image {{
                    max-width: 80%;
                    margin: 0 auto;
                }}
                .banner-text h1, .banner-text p {{
                    max-width: 100%;
                }}
            }}
            /* 7. Atribución en hero */
            .banner-text .attribution {{
                font-size: 0.9rem;
                margin-top: 1rem;
                color: #90A4AE;
            }}
            </style>

            <div class="banner-container">
              <div class="banner-image">
                <img src="data:image/png;base64,{encoded_string}" alt="Portada Rans0mVisor">
              </div>
              <div class="banner-text">
                <h1>Tu radar contra el <span class="accent">Ransomware</span></h1>
                <p>
                  El ransomware es un ataque en el que un programa malicioso encripta tus archivos para luego exigir un rescate.  
                  A diario surgen nuevos grupos con técnicas cada vez más sofisticadas,  
                  mientras las pérdidas económicas y el daño a tu reputación no dejan de crecer.
                </p>
                <!-- Atribución en hero -->
                <em class="attribution"><br>
                  Profundamente agradecidos por la labor de&nbsp;
                  <a href="https://www.ransomware.live" target="_blank" rel="noopener">Ransomware.live</a>
                  al ofrecer datos abiertos que nos ayudan a entender y combatir el ransomware de manera más efectiva.
                </em>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────────
    df = carga_datos_victimas_por_ano()          # Carga datos
    df_filtered = apply_filters(df)              # Aplica filtros

    # ─── Inicializar flag del expander ───────────────────────
    if "expander_open" not in st.session_state:
        st.session_state.expander_open = False

    # ─── Expander en sidebar ───────────────────────────────────
    with st.sidebar.expander("🔄 Actualizar Datos", expanded=st.session_state.expander_open):
        if st.button(" Iniciar actualización", key="home_update"):
            # Abrimos el expander para mostrar la barra de progreso
            st.session_state.expander_open = True
            run_update_flow()

    # ─── Mostrar resumen y cerrar expander ────────────────────
    if st.session_state.show_summary and st.session_state.update_result:
        show_update_summary()
        # Una vez mostrado el resumen, cerramos el expander
        st.session_state.expander_open = False

    # ─── CONTENIDO PRINCIPAL ──────────────────────────────────────────────────────
    st.title("Visión General de Incidentes de Ransomware ")

    eda.show_table_and_metrics(df_filtered)      # Métricas clave y tabla
    eda.plot_interactive_timeline(df)            # Gráfico de dispersión

    # ─── FOOTER ───────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr style="margin-top:3rem; border:none; border-top:1px solid #37474F;">
        <div style="text-align:center; font-size:0.8rem; color:#78909C;">
            Dataset © <a href="https://www.ransomware.live" target="_blank" rel="noopener">
            Ransomware.live</a> — licencia no comercial
        </div>
        """,
        unsafe_allow_html=True
    )
