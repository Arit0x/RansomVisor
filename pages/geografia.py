import streamlit as st
from utils import load_css, carga_datos_victimas_por_ano
from sidebar import apply_filters
import eda
from eda import _geolocate, show_heatmap_pydeck, precalcular_geolocalizaciones
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

    # Interfaz con pestañas para separar el mapa (pesado) del resto del contenido
    tab1, tab2 = st.tabs(["📊 Países más afectados", "🛰️ Mapa de Calor Geoespacial"])

    # Pestaña 1: Países más afectados (carga rápida)
    with tab1:
        # Esta sección carga rápidamente
        eda.plot_paises_mas_afectados(df_filtered)
        
        # Añadir algunas estadísticas adicionales rápidas
        st.subheader("Estadísticas por Región")
        
        # Mostrar el número total de incidentes por país en una tabla
        country_counts = df_filtered['pais'].value_counts().reset_index()
        country_counts.columns = ['País', 'Incidentes']
        st.dataframe(country_counts.head(10), use_container_width=True)

    # Pestaña 2: Mapa de calor (carga diferida mediante botón)
    with tab2:
        st.subheader("🛰️ Mapa de Calor Geoespacial")
        
        # Mostrar botón para cargar el mapa solo cuando se solicite
        if st.button("📍 Cargar Mapa Interactivo", type="primary", use_container_width=True):
            # Al hacer clic, mostrar un spinner mientras se carga
            with st.spinner("Generando mapa de calor..."):
                df_geo = df_filtered[df_filtered['grupo'].isin(groups_sel)].copy()
                
                # Aplicar geolocalización 
                coords = df_geo['pais'].apply(_geolocate)
                df_geo['lat'] = coords.map(lambda x: x[0])
                df_geo['lon'] = coords.map(lambda x: x[1])
                
                if df_geo.dropna(subset=['lat', 'lon']).empty:
                    st.warning("No hay coordenadas válidas para los filtros seleccionados.")
                else:
                    # Mostrar el mapa completo con todos los efectos visuales
                    show_heatmap_pydeck(df_geo)
        else:
            # Si no se ha hecho clic en el botón, mostrar un mensaje informativo
            st.info("👆 Haz clic en el botón 'Cargar Mapa Interactivo' para ver la distribución geográfica de incidentes.")
            
            # Marcador de posición con estilo
            st.markdown("""
            <div style="
                height: 400px;
                background-color: #1E1E1E;
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 20px 0;
                border: 1px dashed #555;
                flex-direction: column;
            ">
                <div style="
                    font-size: 48px;
                    margin-bottom: 20px;
                ">🌍</div>
                <div style="
                    color: #AAA;
                    font-size: 18px;
                    text-align: center;
                    padding: 0 20px;
                ">
                    Vista previa del mapa de incidentes ransomware<br>
                    <span style="font-size: 14px; color: #888;">
                        Haz clic en el botón superior para cargar el mapa interactivo
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


    # ─── FOOTER ───────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <hr style="margin-top:3rem; border:none; border-top:1px solid #37474F;">
        <div style="text-align:center; font-size:0.8rem; color:#78909C;">
            Dataset &copy; <a href="https://www.ransomware.live" target="_blank" rel="noopener">
            Ransomware.live</a> — licencia no comercial
        </div>
        """,
        unsafe_allow_html=True
    )