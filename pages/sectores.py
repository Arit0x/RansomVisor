# pages/sectores.py

import streamlit as st
import pandas as pd

from utils import load_css, carga_datos_victimas_por_ano
from sidebar import apply_filters
import eda


def sectores_app():
    # ─── CARGA DE ESTILOS ─────────────────────────────────────────────────────────
    load_css()
    
    # ─── CARGA DE DATOS ─────────────────────────────────────────────────────────────
    df = carga_datos_victimas_por_ano()
    
    # ─── SIDEBAR: Aplicar filtros ──────────────────────────────────────────────
    df_filtered = apply_filters(df)

    # ─── CONTENIDO PRINCIPAL ────────────────────────────────────────────────────────
    st.title("Dashboard Sectorial 🏭")

    # Obtener el sector seleccionado (si existe)
    sector_filtrado = None
    if 'sidebar_sector' in st.session_state:
        sector_seleccionado = st.session_state.sidebar_sector
        if sector_seleccionado not in ["Todos", "Todos (Sin Desconocidos)"]:
            sector_filtrado = sector_seleccionado

    # 1) VISTA GLOBAL
    if sector_filtrado is None:
        st.subheader("Visión General: Víctimas por Sector")

        # --- Agrupamos por sector y contamos cuántas filas hay en cada uno ---
        totales = df_filtered['sector'].value_counts().sort_values(ascending=False)

        # --- Métrica global ---
        total_global = len(df_filtered)
        st.metric("Total Víctimas (Sectores Filtrados)", total_global)

        # --- Preparamos df_cmp con índice=Sector y columna Total ---
        df_cmp = totales.to_frame(name='Total')
        df_cmp.index.name = 'Sector'

        # 1. Tabla interactiva de sectores
        eda.show_sector_table(df_cmp)

        # 2. Top N sectores
        st.subheader("📈 Top sectores por número de víctimas")
        eda.plot_top_n_sectors_bar(df_cmp)

        # 3. Bubble chart de sectores
        st.subheader("🟢 Bubble chart de sectores")
        eda.plot_sector_bubble(df_cmp)

        # 4. Comparativa intersectorial
        st.subheader("Comparativa Intersectorial")
        eda.plot_intersector_comparison(df_cmp)

        # Para sector
        st.subheader("Análisis por Sector")
        eda.show_sector_analysis(df_filtered)

    # 2) VISTA POR SECTOR ESPECÍFICO
    else:
        st.subheader(f"Análisis de Víctimas en Sector: {sector_filtrado}")

        # Ya tenemos los datos filtrados del sector concreto en df_filtered
        if df_filtered.empty:
            st.warning("No hay datos para el sector seleccionado con los filtros actuales.")
            return

        # Métricas clave y evoluciones
        eda.show_sector_metrics(df_filtered, sector_filtrado)

        st.subheader("Evolución Mensual de Víctimas")
        eda.plot_sector_monthly(df_filtered, sector_filtrado)

        eda.plot_sector_by_group(df_filtered, sector_filtrado, top_n=5)

        st.subheader("Mapa de Víctimas")
        eda.show_sector_geo(df_filtered, sector_filtrado)

        # Dashboard comparativo sector: reutilizamos totales globales
        totales_global = df['sector'].value_counts()
        eda.show_sector_dashboard(
            df_sector=df_filtered,
            sector_es=sector_filtrado,
            totales_global=totales_global.to_dict()
        )

    # ─── FOOTER ───────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr style="margin-top:3rem; border:none; border-top:1px solid #37474F;">
        <div style="text-align:center; font-size:0.8rem; color:#78909C;">
            Dataset  <a href="https://www.ransomware.live" target="_blank" rel="noopener">
            Ransomware.live</a> — licencia no comercial
        </div>
        """,
        unsafe_allow_html=True
    )
