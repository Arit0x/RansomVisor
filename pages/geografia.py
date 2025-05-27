import streamlit as st
from utils import load_css, carga_datos_victimas_por_ano
from sidebar import apply_filters
import eda
from eda import _geolocate, show_heatmap_pydeck
from streamlit_folium import st_folium

def geografia_app():

    # ─── ESTILOS ─────────────────────────────────────────────────────────────────
    load_css()

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────────
    # display_logo()
    # Carga y filtro de datos
    df = carga_datos_victimas_por_ano()
    df_filtered = apply_filters(df)
    # Controles específicos para mapa
    groups_sel = df_filtered['grupo'].unique()

    # ─── CONTENIDO PRINCIPAL ───────────────────────────────────────────────────────
    st.title("Geografía de Incidentes de Ransomware 🌐")

    # Mapa de calor geoespacial
    st.subheader("🛰️ Mapa de Calor Geoespacial")
    df_geo = df_filtered[df_filtered['grupo'].isin(groups_sel)].copy()

    # Aplica geolocalización si aún no lo has hecho
    coords = df_geo['pais'].apply(_geolocate)
    df_geo['lat'] = coords.map(lambda x: x[0])
    df_geo['lon'] = coords.map(lambda x: x[1])

    if df_geo.dropna(subset=['lat', 'lon']).empty:
        st.warning("No hay coordenadas válidas para los filtros seleccionados.")
    else:
        show_heatmap_pydeck(df_geo)

    st.markdown("---")

    # Top países más afectados
    eda.plot_paises_mas_afectados(df_filtered)


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