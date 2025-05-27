from utils import load_css, carga_datos_victimas_por_ano
from sidebar import apply_filters
import eda
import streamlit as st

def tendencias_app():
    # Inyectar estilos globales
    load_css()

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────────
    # Mostrar logo y filtros de datos
    # display_logo()
    df = carga_datos_victimas_por_ano()
    df_filtered = apply_filters(df)

    # ─── CONTENIDO PRINCIPAL ───────────────────────────────────────────────────────
    st.title("Tendencias de Incidentes de Ransomware 📈")

    # Evolución anual
    st.subheader("Incidentes Anuales de Ransomware")
    eda.plot_ataques_simple(df_filtered)

    # Evolución mensual (serie agrupada por mes)
    st.subheader("Incidentes Mensuales por Año")
    eda.plot_evolucion_mensual(df_filtered)

    # Top grupos por año
    st.subheader("Top 10 de Grupos más Activos")
    eda.show_group_analysis(df_filtered)

    # Distribución anual por grupo (stacked area)
    st.subheader("Tendencia de Grupos Recurrentes")
    eda.plot_distribucion_anual_por_grupo(df_filtered)

    # Opcional: recomendamos ir a Home para Forecast/LSTM
    st.markdown("---")
    st.info("Para pronósticos detallados y predicciones, visita la página **Modelado**.")

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
