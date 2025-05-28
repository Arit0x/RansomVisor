import streamlit as st
import pandas as pd
import plotly.express as px
import geonamescache
from streamlit_folium import st_folium
import pydeck as pdk
from folium.plugins import HeatMap
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
import scipy.cluster.hierarchy as sch
from utils import carga_datos_victimas_por_ano, ES_TO_SECTOR

# ─── MAPEOS DE PAÍS ──────────────────────────────────────────────────────────────

SPANISH_TO_ISO = {
    "ESTADOS UNIDOS":      "US", "JAPÓN":             "JP",
    "FRANCIA":             "FR", "REINO UNIDO":       "GB",
    "ALEMANIA":            "DE", "SUIZA":            "CH",
    "AUSTRALIA":           "AU", "CANADÁ":           "CA",
    "PUERTO RICO":         "PR","MONTENEGRO":        "ME",
    "SUECIA":              "SE","IRÁN":              "IR",
    "PAÍSES BAJOS":        "NL","BÉLGICA":          "BE",
    "INDIA":               "IN","EL SALVADOR":       "SV",
    "COLOMBIA":            "CO","PERÚ":             "PE",
    "VENEZUELA":           "VE","URUGUAY":           "UY",
    "CHIPRE":              "CY","ESPAÑA":           "ES",
    "KENIA":               "KE","MALTA":            "MT",
    "ITALIA":              "IT","ETIOPÍA":          "ET",
    "ARGENTINA":           "AR","INDONESIA":        "ID",
    "BRASIL":              "BR","SUDÁFRICA":         "ZA",
    "NUEVA ZELANDA":       "NZ","SINGAPUR":         "SG",
    "HAITÍ":               "HT","PANAMÁ":           "PA",
    "REPÚBLICA CHECA":     "CZ","POLONIA":          "PL",
    "TAIWÁN":              "TW","LUXEMBURGO":       "LU",
    "TRINIDAD Y TOBAGO":   "TT","NORUEGA":          "NO",
    "RUSIA":               "RU","ISLANDIA":         "IS",
    "DINAMARCA":           "DK","MOLDAVIA":          "MD",
    "SRI LANKA":           "LK","FINLANDIA":         "FI",
    "JORDANIA":            "JO","HONG KONG":        "HK",
    "FILIPINAS":           "PH","BERMUDAS":          "BM",
    "ISLAS COOK":          "CK","GRECIA":           "GR",
    "GUATEMALA":           "GT","GUAM":              "GU",
    "CUBA":                "CU","NUEVA CALEDONIA":   "NC",
    "LESOTO":              "LS","SURINAM":          "SR",
    "LITUANIA":            "LT","ANTIGUA Y BARBUDA": "AG",
    "BURUNDI":             "BI","ISLAS VÍRGENES EE.UU.": "VI",
    "CAMERÚN":             "CM","COSTA RICA":        "CR",
    "RUMANÍA":             "RO","MALAWI":            "MW",
    "REUNIÓN":             "RE","REPÚBLICA DOMINICANA": "DO",
    "MALASIA":             "MY","VIETNAM":           "VN",
    "HONDURAS":            "HN","MALÍ":              "ML",
    "CROACIA":             "HR","ECUADOR":           "EC",
    "ISLAS VÍRGENES BRITÁNICAS":"VG","FIYI":           "FJ",
    "MÓNACO":              "MC","KUWAIT":            "KW",
    "MACEDONIA DEL NORTE":"MK","HUNGRÍA":            "HU",
    "BURKINA FASO":        "BF","LETONIA":           "LV",
    "NIGERIA":             "NG","MONGOLIA":          "MN",
    "ISLAS TURCAS Y CAICOS":"TC","BULGARIA":         "BG",
    "BARÉIN":              "BH","ISLAS PITCAIRN":    "PN",
    "NAMIBIA":             "NA","MICRONESIA":        "FM",
    "KOSOVO":              "XK"
}

# (Opcional) si necesitas la inversa:

ISO_TO_SPANISH = {v: k for k, v in SPANISH_TO_ISO.items()}

# ─── INICIALIZAR GEONAMESCACHE ─────────────────────────────────────────────────

_gc        = geonamescache.GeonamesCache()
_COUNTRIES = _gc.get_countries()
_CITIES    = _gc.get_cities()


# ─── FUNCIONES DE GEOLOCALIZACIÓN ──────────────────────────────────────────────
# ─── CACHE PARA GEOLOCALIZACIÓN ─────────────────────────────────────────────────
# Diccionario global para almacenar resultados de geolocalización
_GEO_CACHE = {}
import os
import json

# Ruta al archivo de caché de geolocalizaciones
GEO_CACHE_FILE = os.path.join('data', 'geo_cache.json')

# Función para cargar caché de geolocalizaciones desde archivo
@st.cache_data(ttl=24*3600)
def _load_geo_cache():
    """Carga el caché de geolocalizaciones desde el archivo"""
    global _GEO_CACHE
    try:
        # Asegurar que el directorio data exista
        os.makedirs('data', exist_ok=True)
        
        # Cargar caché desde archivo si existe
        if os.path.exists(GEO_CACHE_FILE):
            with open(GEO_CACHE_FILE, 'r', encoding='utf-8') as f:
                # Convertir las claves a mayúsculas
                cached_data = json.load(f)
                _GEO_CACHE = {k.upper(): v for k, v in cached_data.items()}
                return _GEO_CACHE
    except Exception as e:
        st.warning(f"Error al cargar caché de geolocalizaciones: {e}")
    return {}

# Función para guardar caché de geolocalizaciones a archivo
def _save_geo_cache():
    """Guarda el caché de geolocalizaciones al archivo"""
    try:
        os.makedirs('data', exist_ok=True)
        with open(GEO_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(_GEO_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Error al guardar caché de geolocalizaciones: {e}")

# Cargar caché al inicio
_GEO_CACHE = _load_geo_cache()

@st.cache_data(ttl=24*3600)
def _geolocate(pais: str) -> tuple[float, float] | tuple[None, None]:
    """
    Geolocaliza usando el código ISO-2 si ya es válido.
    Si se proporciona el nombre del país en español, lo traduce primero.
    Usa caché para mejorar el rendimiento.
    """
    if not pais:
        return None, None
    
    clave = pais.strip().upper()
    
    # Verificar caché primero para evitar cálculos repetidos
    if clave in _GEO_CACHE:
        # Convertir a tuple si está almacenado como lista
        result = _GEO_CACHE[clave]
        if isinstance(result, list) and len(result) == 2:
            return tuple(result)
        return result
    
    # Si ya es un código ISO de 2 letras
    if len(clave) == 2 and clave.isalpha():
        iso = clave
    else:
        iso = SPANISH_TO_ISO.get(clave)
        if not iso:
            _GEO_CACHE[clave] = (None, None)
            _save_geo_cache()  # Guardar en archivo
            return None, None
    
    # Intento con la capital
    info = _COUNTRIES.get(iso)
    capital = info.get('capital') if info else None
    if capital:
        for city in _CITIES.values():
            if (city.get('countrycode','').upper() == iso and
                city.get('name','').lower() == capital.lower()):
                try:
                    result = (float(city['latitude']), float(city['longitude']))
                    _GEO_CACHE[clave] = result
                    _save_geo_cache()  # Guardar en archivo
                    return result
                except:
                    break
    
    # Fallback: centroide del país
    coords = [
        (float(c['latitude']), float(c['longitude']))
        for c in _CITIES.values()
        if c.get('countrycode','').upper() == iso
           and c.get('latitude') and c.get('longitude')
    ]
    if coords:
        lats, lons = zip(*coords)
        result = (sum(lats)/len(lats), sum(lons)/len(lons))
        _GEO_CACHE[clave] = result
        _save_geo_cache()  # Guardar en archivo
        return result
    
    _GEO_CACHE[clave] = (None, None)
    _save_geo_cache()  # Guardar en archivo
    return None, None

# Función para geolocalizar países de forma masiva y precalcular
def precalcular_geolocalizaciones(paises):
    """
    Geolocaliza una lista de países de forma masiva y guarda los resultados en caché.
    Útil para precalcular todas las geolocalizaciones necesarias.
    """
    resultados = {}
    for pais in paises:
        if not pais:
            continue
        
        clave = pais.strip().upper()
        if clave not in _GEO_CACHE:
            coords = _geolocate(pais)
            resultados[clave] = coords
    
    # Actualizar la interfaz con el progreso
    if resultados:
        st.success(f"Se han precalculado {len(resultados)} geolocalizaciones nuevas.")
    return resultados

# ─── MÉTRICAS Y TABLA ────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st
import plotly.express as px

def show_table_and_metrics(df: pd.DataFrame): 
    """
    Muestra solamente las columnas esenciales en la tabla y
    calcula métricas sin provocar KeyErrors.
    """
    st.subheader("Tabla de Incidentes Filtrados")

    # Columnas candidatas en orden de preferencia
    posibles_cols = ["grupo", "pais", "victima", "sector", "fecha"]
    # Nos quedamos solo con las que existan
    cols_to_show = [c for c in posibles_cols if c in df.columns]

    # Si no hay ni siquiera fecha, abortamos
    if "fecha" not in cols_to_show:
        st.error("El DataFrame no contiene columna 'fecha'.")
        return

    # Preparamos el DataFrame para mostrar
    df_display = df[cols_to_show].copy()
    # Convertimos 'fecha' a date para no mostrar la hora
    if "fecha" in df_display.columns:
        df_display["fecha"] = pd.to_datetime(df_display["fecha"], errors="coerce").dt.date

    # Mostramos tabla
    st.dataframe(df_display, use_container_width=True)

    total = len(df_display)
    st.subheader("Métricas Generales")
    if total:
        # Aseguramos que 'fecha' esté en datetime en el df de métricas
        df_tmp = df.copy()
        df_tmp["fecha"] = pd.to_datetime(df_tmp["fecha"], errors="coerce")

        primer  = df_tmp["fecha"].min().date()
        ultimo  = df_tmp["fecha"].max().date()
        dias    = (ultimo - primer).days + 1
        promedio = total / dias

        last_week = (
            df_tmp
            .set_index("fecha")
            .last("7D")
            .resample("D")
            .size()
        )
        prev_week = (
            df_tmp
            .set_index("fecha")
            .last("14D")
            .head(7)
            .resample("D")
            .size()
        )
        avg_last = last_week.mean()
        avg_prev = prev_week.mean()
        delta    = avg_last - avg_prev

        # Métricas principales
        c1, c2, c3, = st.columns(3)
        c1.metric("Total incidentes", total)
        c2.metric("Período", f"{primer} → {ultimo}")
        c3.metric("Días analizados", dias)

        # Métricas adicionales: total de grupos, países y sectores
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Promedio diario", f"{promedio:.2f}", delta=f"{delta:.2f}")
        c6.metric("Total grupos", df_display["grupo"].nunique() if "grupo" in df_display.columns else 0)
        c7.metric("Total países", df_display["pais"].nunique() if "pais" in df_display.columns else 0)
        c8.metric("Total sectores", df_display["sector"].nunique() if "sector" in df_display.columns else 0)

        # Top 5 grupos y países
        top_grupos = (
            df_display.get("grupo", pd.Series(dtype=object))
            .value_counts()
            .head(5)
            .rename_axis("Grupo")
            .reset_index(name="Incidentes")
        )
        top_paises = (
            df_display.get("pais", pd.Series(dtype=object))
            .value_counts()
            .head(5)
            .rename_axis("País")
            .reset_index(name="Incidentes")
        )

        g1, g2 = st.columns(2)
        g1.subheader("Top 5 Grupos")
        g1.dataframe(top_grupos, use_container_width=True)
        g2.subheader("Top 5 Países")
        g2.dataframe(top_paises, use_container_width=True)

        st.subheader("Incidentes por Día de la Semana")
        df_tmp["weekday"] = df_tmp["fecha"].dt.day_name()
        week_counts = df_tmp["weekday"].value_counts().reindex(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        )

        fig2 = px.bar(
            x=week_counts.index,
            y=week_counts.values,
            labels={"x":"Día","y":"Incidentes"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No hay datos para mostrar métricas.")




# ─── GRÁFICOS SIMPLES STREAMLIT ─────────────────────────────────────────────────

def plot_incidentes_por_grupo(df):
    st.subheader("Incidentes por Grupo")
    st.bar_chart(df['grupo'].value_counts())

def plot_serie_temporal(df):
    """
    Serie temporal diaria con media móvil y selector de rango.
    """
    df_copy = df.copy()
    df_copy['fecha'] = pd.to_datetime(df_copy['fecha'])
    daily = df_copy.set_index('fecha').resample('D').size().rename('incidentes')
    rolling7 = daily.rolling(7).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values,
                             mode='lines', name='Incidentes diarios', hovertemplate='%{y}'))
    fig.add_trace(go.Scatter(x=rolling7.index, y=rolling7.values,
                             mode='lines', line=dict(dash='dash'),
                             name='Media móvil (7d)'))
    fig.update_layout(
        # title=''
        xaxis_title='Fecha', yaxis_title='Número de incidentes',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_ataques_simple(df):
    """
    Gráfico sencillo: barras de incidentes por año,
    línea de media móvil y anotaciones de hitos.
    """
    # Preparar datos
    df['año'] = df['fecha'].dt.year
    yearly = df.groupby('año').size().rename('incidentes')
    rolling = yearly.rolling(3, center=True).mean()

    # Base: barras de incidentes
    fig = px.bar(
        x=yearly.index, y=yearly.values,
        labels={'x':'Año', 'y':'Incidentes'}
        # title=
    )
    fig.update_traces(marker_color='steelblue', opacity=0.8)

    # Línea de media móvil
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling.values,
        mode='lines', name='Media móvil (3 años)',
        line=dict(color='crimson', width=3)
    ))

    # Anotaciones de hitos
    milestones = {2019: 'REvil activo', 2021: 'LockBit 2.0'}
    for año, texto in milestones.items():
        if año in yearly.index:
            fig.add_annotation(
                x=año, y=yearly[año] + max(yearly)*0.05,
                text=texto,
                showarrow=True, arrowhead=2, ax=0, ay=-20
            )

    # Layout limpio
    fig.update_layout(
        template='simple_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickmode='linear'),
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_evolucion_mensual(df: pd.DataFrame):
    """
    Gráficas de evolución mensual: barras comparativas por año y línea de tendencia.
    """
    # Aseguramos datetime
    df_tmp = df.copy()
    df_tmp['fecha'] = pd.to_datetime(df_tmp['fecha'], errors='coerce')

    # Conteo mensual
    monthly = (
        df_tmp
        .resample('M', on='fecha')
        .size()
        .reset_index(name='incidentes')
    )
    monthly['año']   = monthly['fecha'].dt.year
    monthly['mes']   = monthly['fecha'].dt.strftime('%b')
    # Orden de meses breve en inglés o adapta a español:
    meses_orden = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    monthly['mes'] = pd.Categorical(monthly['mes'], meses_orden)

    # 1) Gráfico de barras: comparación mes a mes por año
    fig_bar = px.bar(
        monthly,
        x='mes',
        y='incidentes',
        color='año',
        barmode='group',
        #title=
        labels={'mes':'Mes', 'incidentes':'Incidentes', 'año':'Año'}
    )
    fig_bar.update_layout(template='plotly_white')
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2) Línea de tendencia: serie mensual con marcadores
    fig_line = px.line(
        monthly,
        x='fecha',
        y='incidentes',
        title='Tendencia Mensual de Incidentes',
        markers=True,
        labels={'fecha':'Fecha','incidentes':'Incidentes'}
    )
    fig_line.update_layout(template='plotly_white')
    st.plotly_chart(fig_line, use_container_width=True)


def plot_distribucion_anual_por_grupo(df: pd.DataFrame, top_n: int = 10):
    """
    Líneas individuales para los top N grupos + 'Otros', mostrando su evolución anual.
    """
    df_tmp = df.copy()
    df_tmp['año'] = pd.to_datetime(df_tmp['fecha'], errors='coerce').dt.year

    # Pivot: conteo por año y grupo
    pivot = (
        df_tmp
        .pivot_table(index='año', columns='grupo', aggfunc='size', fill_value=0)
    )

    # Seleccionamos los top N grupos por total de incidentes
    totales = pivot.sum().sort_values(ascending=False)
    grupos_top = list(totales.head(top_n).index)

    # Construimos DataFrame con top N y agregamos 'Otros'
    pivot_top = pivot[grupos_top].copy()
    pivot_top['Otros'] = pivot.drop(columns=grupos_top, errors='ignore').sum(axis=1)

    # Pasamos a formato largo para plotly
    df_long = (
        pivot_top
        .reset_index()
        .melt(id_vars='año', var_name='grupo', value_name='incidentes')
    )

    # Gráfico de líneas
    fig = px.line(
        df_long,
        x='año',
        y='incidentes',
        color='grupo',
        markers=True,
        # title=f'Evolución Anual por Grupo (Top {top_n} + Otros)
        labels={'año':'Año','incidentes':'Incidentes','grupo':'Grupo'}
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)


def plot_interactive_timeline(df: pd.DataFrame) -> None:
    """
    Muestra una línea temporal animada que visualiza la evolución de incidentes a lo largo del tiempo.
    """
    st.subheader("🎬 Línea Temporal Animada de Incidentes")

    if df.empty or 'fecha' not in df.columns:
        st.warning("No hay datos con fecha para dibujar la línea temporal animada.")
        return

    # Preparar fechas
    df = df.dropna(subset=['fecha']).copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    # Agrupar por semana
    weekly = (
        df
        .assign(
            semana=df['fecha']
                .dt.to_period('W')
                .apply(lambda r: r.start_time)
        )
        .groupby('semana')
        .size()
        .reset_index(name='incidentes')
        .sort_values('semana')
    )

    if weekly.empty:
        st.warning("Después del agrupado por semana no queda ningún registro.")
        return

    # Añadir columna acumulativa para mayor impacto visual
    weekly['incidentes_acumulados'] = weekly['incidentes'].cumsum()
    
    # Crear pestañas para diferentes visualizaciones
    tab1, tab2 = st.tabs(["🎭 Animación", "📊 Exploración"])
    
    with tab1:
        # Controles de animación
        st.write("### Controles de Animación")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            velocidad = st.select_slider(
                "Velocidad de animación",
                options=["Muy lenta", "Lenta", "Media", "Rápida", "Muy rápida"],
                value="Media"
            )
        with col2:
            mostrar_acumulados = st.checkbox("Mostrar valores acumulados", value=False)
        with col3:
            tema_color = st.selectbox(
                "Tema de color",
                ["Moderno", "Futurista", "Neón", "Corporativo", "Elegante"]
            )
        
        # Mapear selecciones a valores
        duracion_frames = {
            "Muy lenta": 1000, 
            "Lenta": 700, 
            "Media": 500, 
            "Rápida": 300, 
            "Muy rápida": 100
        }
        
        # Esquemas de colores según tema
        paletas = {
            "Moderno": ["#0099ff", "#00ccff", "#ff6600", "#FFD700"],
            "Futurista": ["#00ffcc", "#33ccff", "#cc00ff", "#00ff99"],
            "Neón": ["#ff00ff", "#00ffff", "#ffff00", "#ff3399"],
            "Corporativo": ["#003366", "#0066cc", "#ff9900", "#33cc33"],
            "Elegante": ["#4B0082", "#8A2BE2", "#9370DB", "#9932CC"]
        }
        
        colores = paletas[tema_color]
        
        # Crear datos para animación - una fila por cada punto en el tiempo
        y_data = 'incidentes_acumulados' if mostrar_acumulados else 'incidentes'
        max_y = weekly[y_data].max() * 1.1  # Para escala del eje Y
        
        # Crear animación con Plotly
        fig = px.line(
            weekly,
            x='semana',
            y=y_data,
            labels={
                'semana': 'Fecha',
                'incidentes': 'Incidentes semanales',
                'incidentes_acumulados': 'Incidentes acumulados'
            },
            title=f"{'Incidentes Acumulados' if mostrar_acumulados else 'Incidentes Semanales'} a lo largo del tiempo",
            range_y=[0, max_y]
        )
        
        # Convertir a formato animado
        frames = []
        for i in range(1, len(weekly) + 1):
            subset = weekly.iloc[:i]
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=subset['semana'],
                        y=subset[y_data],
                        mode='lines+markers',
                        line=dict(width=6, color=colores[0]),
                        marker=dict(size=10, color=colores[1], line=dict(width=2, color=colores[2])),
                        fill='tozeroy',
                        fillcolor=f"rgba({int(colores[3][1:3], 16)}, {int(colores[3][3:5], 16)}, {int(colores[3][5:7], 16)}, 0.2)"
                    )
                ],
                name=f"frame{i}"
            )
            frames.append(frame)
        
        fig.frames = frames
        
        # Añadir controles de animación
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='▶️',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=duracion_frames[velocidad], redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=300, easing='cubic-in-out')
                                )
                            ]
                        ),
                        dict(
                            label='⏸️',
                            method='animate',
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ]
                        ),
                        dict(
                            label='⏮️',
                            method='animate',
                            args=[
                                ['frame1'],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ],
                    x=-0.04,  # Posicionado en la parte inferior izquierda
                    y=-0.20,  # Bien abajo para evitar solapamiento con el gráfico
                    xanchor='left',  # Anclado a la izquierda
                    yanchor='top',
                    direction='right',
                    pad=dict(t=10, r=10),
                    font=dict(size=16, family='Arial Black'),
                    bgcolor='rgba(20, 24, 35, 0.7)',  # Fondo semitransparente que combine con el tema oscuro
                    bordercolor='rgba(255, 255, 255, 0.3)'  # Borde sutil
                )
            ],
            # Añadir slider para la animación
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            method='animate',
                            args=[
                                [f'frame{k+1}'],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ],
                            label=f"{weekly['semana'].iloc[k].strftime('%Y-%m-%d')}"
                        )
                        for k in range(0, len(weekly), max(1, len(weekly) // 10))  # Mostrar ~10 etiquetas
                    ],
                    transition=dict(duration=0),
                    x=0.08,  # Alineado con los botones en la parte inferior izquierda
                    y=-0.05,  # Posicionado justo arriba de los botones
                    len=0.8,  # Ancho del 80% para dejar espacio en los extremos
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix='📅 Fecha: ',
                        visible=True,
                        xanchor='left'  # Alineado a la izquierda
                    ),
                    pad=dict(b=10, t=50),  # Espacio adicional para no interferir con el gráfico
                    bgcolor='rgba(20, 24, 35, 0.3)'  # Fondo semitransparente
                )
            ]
        )
        
        # Añadir efectos visuales y estilo moderno
        fig.update_traces(
            hoverinfo='x+y',
            hovertemplate='<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Incidentes:</b> %{y}<extra></extra>',
        )
        
        # Configurar diseño moderno
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            font=dict(family='Arial, sans-serif', size=14, color='white'),
            title=dict(
                font=dict(size=24, family='Arial Black', color='white'),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showline=True,
                linewidth=2,
                linecolor=colores[0],
                tickformat='%b %Y',
                title=dict(font=dict(size=16, family='Arial, sans-serif')),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                showline=True,
                linewidth=2,
                linecolor=colores[0],
                title=dict(font=dict(size=16, family='Arial, sans-serif')),
                tickfont=dict(size=12)
            ),
            margin=dict(l=40, r=20, t=100, b=140),  # Aumentado aún más el margen inferior para los controles
            height=600,
            hovermode='x unified',
            legend=dict(
                orientation='h',
                y=-0.3  # Ajustado para no solaparse con los controles
            )
        )
        
        # Añadir efectos adicionales
        if tema_color == "Futurista" or tema_color == "Neón":
            fig.add_annotation(
                text="EVOLUCIÓN TEMPORAL",
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28, family='Arial Black', color=colores[1]),
                opacity=0.7
            )
        
        # Mostrar el gráfico animado
        st.plotly_chart(fig, use_container_width=True)
        
        # Añadir estadísticas bajo el gráfico
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total de incidentes",
                f"{weekly['incidentes'].sum():,}",
                delta=None,
                delta_color="normal"
            )
        with col2:
            st.metric(
                "Semana con más incidentes",
                f"{weekly['incidentes'].max():,}",
                f"Semana del {weekly.loc[weekly['incidentes'].idxmax(), 'semana'].strftime('%d/%m/%Y')}"
            )
        with col3:
            st.metric(
                "Promedio semanal",
                f"{weekly['incidentes'].mean():.1f}",
                delta=None
            )
        with col4:
            st.metric(
                "Tendencia último mes",
                f"{(weekly['incidentes'].iloc[-1] / weekly['incidentes'].iloc[-5] - 1) * 100:.1f}%",
                delta="respecto al mes anterior"
            )
    
    with tab2:
        # Pestaña de exploración con diferentes visualizaciones
        st.write("### Exploración detallada")
        
        # Selector de visualización
        viz_tipo = st.radio(
            "Selecciona tipo de visualización",
            ["Línea con área", "Barras"],
            horizontal=True
        )
        
        if viz_tipo == "Línea con área":
            # Crear gráfico de área
            fig2 = px.area(
                weekly,
                x='semana',
                y='incidentes',
                labels={'semana':'Fecha', 'incidentes':'Incidentes'},
                title="Evolución de incidentes a lo largo del tiempo",
                color_discrete_sequence=[colores[0]]
            )
            
            # Personalizar
            fig2.update_traces(
                line=dict(width=3, color=colores[1]),
                fillcolor=f"rgba({int(colores[0][1:3], 16)}, {int(colores[0][3:5], 16)}, {int(colores[0][5:7], 16)}, 0.3)"
            )
            
        elif viz_tipo == "Barras":
            # Crear gráfico de barras
            fig2 = px.bar(
                weekly,
                x='semana',
                y='incidentes',
                labels={'semana':'Fecha', 'incidentes':'Incidentes'},
                title="Incidentes semanales",
                color_discrete_sequence=[colores[2]]
            )
            
            # Añadir línea de tendencia
            fig2.add_trace(
                go.Scatter(
                    x=weekly['semana'],
                    y=weekly['incidentes'].rolling(window=4).mean(),
                    mode='lines',
                    name='Media móvil (4 semanas)',
                    line=dict(color=colores[1], width=4, dash='dot')
                )
            )
            
        elif viz_tipo == "Tendencia & Estacionalidad":
            # Implementar descomposición de series temporales
            try:
                # Crear índice regular para la serie temporal
                ts = weekly.set_index('semana')['incidentes']
                
                # Asegurar que el índice es regular
                ts = ts.asfreq('W')
                
                # Rellenar valores faltantes si los hay
                if ts.isna().any():
                    ts = ts.fillna(method='ffill')
                
                # Descomposición solo si hay suficientes datos
                if len(ts) >= 12:  # Al menos 3 meses de datos
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomposition = seasonal_decompose(ts, model='additive', period=4)
                    
                    # Crear subplots para cada componente
                    fig2 = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=("Datos originales", "Tendencia", "Estacionalidad", "Residuos"),
                        vertical_spacing=0.1
                    )
                    
                    # Añadir cada componente
                    fig2.add_trace(
                        go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Original', 
                                  line=dict(color=colores[0], width=2)),
                        row=1, col=1
                    )
                    
                    fig2.add_trace(
                        go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                                  mode='lines', name='Tendencia',
                                  line=dict(color=colores[1], width=3)),
                        row=2, col=1
                    )
                    
                    fig2.add_trace(
                        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                                  mode='lines', name='Estacionalidad',
                                  line=dict(color=colores[2], width=2)),
                        row=3, col=1
                    )
                    
                    fig2.add_trace(
                        go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                                  mode='lines', name='Residuos',
                                  line=dict(color=colores[3], width=1.5)),
                        row=4, col=1
                    )
                    
                    fig2.update_layout(height=800, title_text="Descomposición de la serie temporal")
                else:
                    st.warning("Se necesitan al menos 12 semanas de datos para la descomposición estacional.")
                    # Mostrar gráfico simple en su lugar
                    fig2 = px.line(
                        weekly, 
                        x='semana', 
                        y='incidentes',
                        title="Serie temporal (datos insuficientes para descomposición)"
                    )
            except Exception as e:
                st.error(f"Error en la descomposición temporal: {e}")
                fig2 = px.line(weekly, x='semana', y='incidentes')
            
        else:  # Mapa de calor
            # Crear datos para el mapa de calor
            weekly['año'] = weekly['semana'].dt.year
            weekly['mes'] = weekly['semana'].dt.month
            weekly['semana_del_mes'] = weekly['semana'].dt.day.apply(lambda x: (x-1)//7 + 1)
            
            # Agregar por mes y semana del mes
            heatmap_data = weekly.pivot_table(
                index='mes',
                columns='semana_del_mes',
                values='incidentes',
                aggfunc='sum'
            ).fillna(0)
            
            # Crear mapa de calor
            fig2 = px.imshow(
                heatmap_data,
                labels=dict(x="Semana del mes", y="Mes", color="Incidentes"),
                x=[f"Semana {i}" for i in heatmap_data.columns],
                y=[calendar.month_name[i] for i in heatmap_data.index],
                color_continuous_scale=[colores[0], colores[3]],
                title="Distribución de incidentes por mes y semana"
            )
            
            fig2.update_layout(
                coloraxis_colorbar=dict(
                    title="Incidentes",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300
                )
            )
        
        # Aplicar estilo común para todos los gráficos
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.02)',
            font=dict(family='Arial, sans-serif', size=12, color='white'),
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        # Mostrar gráfico exploratorio
        st.plotly_chart(fig2, use_container_width=True)
        
        # Mostrar tabla con detalles
        with st.expander("Ver datos detallados"):
            st.dataframe(
                weekly[['semana', 'incidentes', 'incidentes_acumulados']]
                .style.background_gradient(cmap='Blues', subset=['incidentes'])
                .format({'incidentes': '{:.0f}', 'incidentes_acumulados': '{:.0f}'})
            )



# ─── HEATMAP GEOSPACIAL ─────────────────────────────────────────────────────────

# 🔥 Cacheamos el heatmap para 1 hora (3600 segundos)
@st.cache_data(ttl=3600)
def show_heatmap_pydeck(df: pd.DataFrame):
    """
    Muestra un mapa de calor estilo hacker usando Pydeck.
    Espera un DataFrame con columnas 'lat' y 'lon'.
    """
    # Verificamos columnas necesarias antes de procesar
    if 'lat' not in df.columns or 'lon' not in df.columns:
        st.warning("El DataFrame debe contener columnas 'lat' y 'lon'.")
        return
    
    # Optimización: Filtrar datos inválidos directamente sin crear copia
    df_valid = df.dropna(subset=['lat', 'lon'])
    
    # Validación adicional: Verificar que las coordenadas sean numéricas
    try:
        df_valid = df_valid[pd.to_numeric(df_valid['lat'], errors='coerce').notna() &
                            pd.to_numeric(df_valid['lon'], errors='coerce').notna()]
    except Exception:
        st.warning("Las coordenadas deben ser valores numéricos.")
        return
    
    if df_valid.empty:
        st.warning("No hay coordenadas válidas para mostrar el mapa.")
        return
    
    # Optimización: Convertir a tipos numéricos explícitamente para evitar errores
    df_valid['lat'] = pd.to_numeric(df_valid['lat'])
    df_valid['lon'] = pd.to_numeric(df_valid['lon'])
    
    # Optimización: Limitar la cantidad de puntos si es muy grande
    max_points = 10000  # Límite razonable para rendimiento
    if len(df_valid) > max_points:
        df_valid = df_valid.sample(max_points, random_state=42)
    
    # Calcular estadísticas del mapa una sola vez
    mean_lat = df_valid['lat'].mean()
    mean_lon = df_valid['lon'].mean()
    
    # Crear el layer con configuración optimizada
    layer = pdk.Layer(
        "HexagonLayer",
        data=df_valid,
        get_position='[lon, lat]',
        radius=100000,
        extruded=True,
        pickable=True,
        auto_highlight=True,
        # Optimización: Añadir parámetros para mejor rendimiento
        elevation_scale=4,
        coverage=0.9
    )
    
    # Vista inicial del mapa
    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=2,
        pitch=45,
    )
    
    tooltip = {
        "html": """
        <div style='
            font-family: monospace;
            color: #39FF14;
            background-color: black;
            padding: 8px;
            border: 1px solid #39FF14;
            border-radius: 4px;
        '>
            💥 <b>Total de ataques:</b> {elevationValue}
        </div>
        """,
        "style": {
            "backgroundColor": "black",
            "border": "1px solid #39FF14"
        }
    }
    
    try:
        # Mapa estilo oscuro y elegante
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip=tooltip
        )
        
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as e:
        st.error(f"Error al renderizar el mapa: {str(e)}")

# ─── DASHBOARD SECTORIAL ────────────────────────────────────────────────────────

def show_sector_dashboard(df_sector, sector_es, totales_global):
    st.subheader(f"📊 Víctimas en {sector_es}")
    total=len(df_sector)
    st.metric("Total víctimas", total)

    # Evolución mensual
    s = (df_sector.assign(mes=pd.to_datetime(df_sector['fecha']).dt.to_period('M').dt.to_timestamp())
                      .groupby('mes').size())
    st.line_chart(s)

    # Crecimiento interanual
    años = df_sector.groupby(df_sector['fecha'].dt.year).size()
    if len(años)>1:
        prev, curr = años.iloc[-2], años.iloc[-1]
        pct=(curr-prev)/prev*100 if prev else 0
        st.metric(f"Crecimiento {años.index[-2]}→{años.index[-1]}", f"{pct:.1f}%")

    # Comparativa intersectorial
    df_cmp=pd.DataFrame({'Sector':list(totales_global),'Total':list(totales_global.values())}).set_index('Sector')
    df_cmp.at[sector_es,'Total']=total
    fig=px.bar(df_cmp.reset_index(), x='Sector', y='Total', title="Comparativa Sectores")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ─── MÉTRICAS SECTOR SPECÍFICAS ─────────────────────────────────────────────────

def show_sector_metrics(df, sector_es):
    st.subheader(f"Métricas Sector: {sector_es}")
    df['fecha'] = pd.to_datetime(df['fecha'],errors='coerce')
    total=len(df)
    first, last = (df['fecha'].min().date(), df['fecha'].max().date()) if total else (None,None)
    days=(last-first).days+1 if total else None
    avg=total/days if days>0 else None
    c1,c2,c3 = st.columns(3)
    c1.metric("Total víctimas",total)
    c2.metric("Período",f"{first} → {last}" if total else "—")
    c3.metric("Promedio diario",f"{avg:.2f}" if avg else "—")

def plot_sector_monthly(df, sector_es):
    df2=(df.assign(mes=pd.to_datetime(df['fecha']).dt.to_period('M').dt.to_timestamp()).groupby('mes').size().reset_index(name='Víctimas'))
    fig=px.line(df2,x='mes',y='Víctimas')
    st.plotly_chart(fig, use_container_width=True)

def plot_sector_by_group(df, sector_es, top_n=8):
    st.subheader(f"Top {top_n} Grupos dentro del sector: {sector_es}")
    top=df['grupo'].value_counts().head(top_n).rename_axis('Grupo').reset_index(name='Víctimas')
    fig=px.bar(top, x='Víctimas', y='Grupo', orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def show_sector_table(df_cmp: pd.DataFrame):
    """
    Muestra una tabla interactiva de totales por sector,
    con búsqueda y ordenamiento en el propio dataframe.
    """
    st.subheader("🔍 Tabla interactiva de sectores")
    df_display = (
        df_cmp
        .reset_index()
        .rename(columns={'index': 'Sector'})
    )
    st.dataframe(df_display, use_container_width=True)

# ─── Tabla interactiva de sectores ───────────────────────────────────────────────
def show_sector_table(df_cmp: pd.DataFrame):
    """
    Muestra una tabla interactiva de totales por sector,
    con búsqueda y ordenamiento en el propio dataframe.
    """
    st.subheader("🔍 Tabla interactiva de sectores")
    df_display = (
        df_cmp
        .reset_index()
        .rename(columns={'index': 'Sector'})
    )
    st.dataframe(df_display, use_container_width=True)


# ─── Top N sectores (barra horizontal) ───────────────────────────────────────────
def plot_top_n_sectors_bar(df_cmp: pd.DataFrame, default_n: int = 10):
    """
    Gráfico de barras horizontales interactivo para los top N sectores.
    El usuario puede elegir cuántos sectores mostrar con un slider.
    """
    max_n = len(df_cmp)
    top_n = st.slider(
        "Número de sectores a mostrar:",
        min_value=5,
        max_value=max_n,
        value=min(default_n, max_n)
    )
    df_sorted = df_cmp.sort_values('Total', ascending=False).head(top_n)
    df_sorted = df_sorted.reset_index().rename(columns={'index': 'Sector'})

    fig = px.bar(
        df_sorted,
        x='Total',
        y='Sector',
        orientation='h',
        text='Total',
        title=f"Top {top_n} sectores más afectados",
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Número de víctimas',
        yaxis_title='Sector',
        height=400 + 20 * top_n
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Bubble chart de sectores ────────────────────────────────────────────────────
def plot_sector_bubble(df_cmp: pd.DataFrame):
    """
    Bubble chart para visualizar todos los sectores.
    El tamaño de la burbuja refleja el número de víctimas.
    """
    df_plot = df_cmp.reset_index().rename(columns={'index': 'Sector'})
    fig = px.scatter(
        df_plot,
        x='Sector',
        y='Total',
        size='Total',
        hover_name='Sector',
        title='Visualización de sectores mediante bubble chart'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title='Sector',
        yaxis_title='Número de víctimas',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Comparativa intersectorial ────────────────────────────────────────────────
def plot_intersector_comparison(df_cmp: pd.DataFrame):
    """
    Comparativa intersectorial global y año a año en pestañas.

    Parámetro:
      - df_cmp: DataFrame con índice 'Sector' y columna 'Total'.
    """
    # --- Preparar el DataFrame global ---
    df_global = (
        df_cmp
        .reset_index()
        .rename(columns={'index': 'Sector', 'Total': 'Víctimas'})
    )

    # --- Cargar datos anuales desde el snapshot completo ---
    df_all = carga_datos_victimas_por_ano()
    if df_all.empty:
        st.warning("No hay datos históricos de víctimas por año.")
        # Mostramos solo la pestaña global
        tab1, _ = st.tabs(["Global", "Año a año"])
        with tab1:
            st.subheader("Víctimas por Sector (global)")
            total_v = int(df_global['Víctimas'].sum())
            st.metric("Total víctimas (todos los sectores)", total_v)
            fig1 = px.bar(
                df_global,
                x='Sector',
                y='Víctimas',
                title="Víctimas por Sector (global)",
                labels={'Víctimas':'Número de víctimas'}
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
        return

    # Asegurarnos de tener columna 'sector' (en español ya normalizado) y fecha
    df_all = df_all.copy()
    df_all['Year'] = df_all['fecha'].dt.year

    # Agrupar por sector y año
    df_yearly = (
        df_all
        .groupby(['sector', 'Year'])
        .size()
        .reset_index(name='Víctimas')
        .rename(columns={'sector': 'Sector'})
    )

    # Filtrar solo los sectores que aparecen en la comparativa global
    sectores_interes = set(df_global['Sector'])
    df_yearly = df_yearly[df_yearly['Sector'].isin(sectores_interes)]

    # --- Ahora las dos pestañas ---
    tab1, tab2 = st.tabs(["Global", "Año a año"])

    with tab1:
        st.subheader("Víctimas por Sector (global)")
        total_v = int(df_global['Víctimas'].sum())
        st.metric("Total víctimas (todos los sectores)", total_v)

        fig1 = px.bar(
            df_global,
            x='Sector',
            y='Víctimas',
            title="Víctimas por Sector (global)",
            labels={'Víctimas':'Número de víctimas'}
        )
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Comparativa intersectorial año a año")
        if not df_yearly.empty:
            fig2 = px.bar(
                df_yearly,
                x='Year',
                y='Víctimas',
                color='Sector',
                barmode='group',
                title="Víctimas por Sector año a año",
                labels={'Year':'Año','Víctimas':'Número de víctimas'}
            )
            fig2.update_layout(xaxis_tickangle=-45, legend_title_text='Sector')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay datos suficientes para la vista Año a año.")

# ─── MAPEOS ─────────────────────────────────────────────────────────────────────

# Cache para el Deck de círculos
@st.cache_resource
def _build_circles_deck(circles: pd.DataFrame, zoom: int = 2) -> pdk.Deck:
    centro = [circles['lat'].mean(), circles['lon'].mean()]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=circles,
        get_position=["lon","lat"],
        get_radius="radius",
        get_fill_color=[0,128,200,160],
        pickable=True,
        tooltip={"html":"<b>{iso}</b>: {count} víctimas"}
    )
    view = pdk.ViewState(latitude=centro[0], longitude=centro[1], zoom=zoom)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="mapbox://styles/mapbox/light-v9"
    )

def show_sector_geo(df: pd.DataFrame, sector_es: str):
    st.subheader(f"Distribución Geográfica: {sector_es}")
    df = df.copy()

    # 1) Si ya tienes lat/lon individuales, dibuja puntos rojos sobre fondo oscuro
    if {'lat','lon'}.issubset(df.columns):
        pts = df.dropna(subset=['lat','lon'])
        if not pts.empty:
            centro = [pts['lat'].mean(), pts['lon'].mean()]
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=pts,
                get_position=["lon","lat"],
                get_radius=30000,
                # **ROJO INTENSO con algo de transparencia**
                get_fill_color=[255, 0, 0, 200],
                pickable=True,
            )
            view = pdk.ViewState(latitude=centro[0], longitude=centro[1], zoom=2)
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                # **ESTILO OSCURO DE MAPBOX**
                map_style="mapbox://styles/mapbox/dark-v10"
            )
            st.pydeck_chart(deck, use_container_width=True)
            return

    # 2) Fallback: agrupar por país y dibujar círculos proporcionales
    cuentas = df['pais'].str.upper().value_counts()
    if cuentas.empty:
        st.warning("No hay datos de país para geolocalizar.")
        return

    rows = []
    for iso, cnt in cuentas.items():
        lat, lon = _geolocate(iso)
        if lat is not None:
            rows.append({"iso":iso, "count":cnt, "lat":lat, "lon":lon})
    circles = pd.DataFrame(rows)

    if circles.empty:
        # Tabla de fallback (sin cambios)
        df_tab = cuentas.rename_axis("ISO").reset_index(name="Víctimas")
        df_tab["País"] = df_tab["ISO"].map(ISO_TO_SPANISH).fillna(df_tab["ISO"])
        st.dataframe(df_tab[["País","Víctimas"]], use_container_width=True)
        return

    # Calcular radios       
    maxc = circles['count'].max()
    circles['radius'] = circles['count']/maxc*50000 + 20000

    # 3) Dibujar círculos rojos sobre fondo oscuro
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=circles,
        get_position=["lon","lat"],
        get_radius="radius",
        # **ROJO MÁS SUAVE y ligeramente transparente**
        get_fill_color=[200, 0, 0, 180],
        pickable=True,
        tooltip={"html":"<b>{iso}</b>: {count} víctimas"}
    )
    centro = [circles['lat'].mean(), circles['lon'].mean()]
    view = pdk.ViewState(latitude=centro[0], longitude=centro[1], zoom=2)
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        # **MISMO ESTILO OSCURO**
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    st.pydeck_chart(deck, use_container_width=True)


def analisis_integrado():
    """
    Realiza un análisis completo utilizando datos integrados de múltiples fuentes.
    Muestra visualizaciones y métricas combinadas.
    """
    st.title("Análisis Integrado de Ciberataques")
    
    # Cargar datos integrados
    with st.spinner("Cargando datos integrados..."):
        df_integrado, metricas, dfs_originales = integrar_datos_completos()
    
    if df_integrado.empty:
        st.warning("No hay datos disponibles para el análisis integrado.")
        return
    
    # Mostrar métricas generales
    st.subheader("Métricas Globales")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Registros", f"{metricas['total_registros']:,}")
        st.metric("Víctimas Únicas", f"{metricas['total_victimas_unicas']:,}")
    
    with col2:
        st.metric("Grupos Detectados", f"{metricas['total_grupos']:,}")
        st.metric("Países Afectados", f"{metricas['total_paises']:,}")
    
    with col3:
        st.metric("Sectores Analizados", f"{metricas['total_sectores']:,}")
        if metricas['rango_años'][0] and metricas['rango_años'][1]:
            st.metric("Período", f"{metricas['rango_años'][0]} - {metricas['rango_años'][1]}")
    
    # Mostrar fuentes de datos utilizadas
    st.subheader("Fuentes de Datos Utilizadas")
    for fuente in metricas['fuentes_datos']:
        st.markdown(f"- {fuente}")
    
    # Análisis por año (combinando todas las fuentes)
    st.subheader("Evolución Temporal")
    
    if 'año' in df_integrado.columns:
        # Agrupar por año y fuente
        df_anual = df_integrado.groupby(['año', 'fuente']).size().reset_index(name='cantidad')
        
        # Crear gráfico
        fig = px.line(df_anual, x='año', y='cantidad', color='fuente',
                     title="Evolución anual por fuente de datos",
                     labels={'año': 'Año', 'cantidad': 'Cantidad', 'fuente': 'Fuente de datos'})
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución por año (todas las fuentes combinadas)
        df_anual_total = df_integrado.groupby('año').size().reset_index(name='cantidad')
        fig2 = px.bar(df_anual_total, x='año', y='cantidad',
                     title="Distribución anual (todas las fuentes)",
                     labels={'año': 'Año', 'cantidad': 'Cantidad'})
        fig2.update_layout(template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Análisis por grupo (top 10)
    if 'grupo' in df_integrado.columns:
        st.subheader("Análisis por Grupo")
        
        # Top 10 grupos
        top_grupos = df_integrado['grupo'].value_counts().head(10).reset_index()
        top_grupos.columns = ['Grupo', 'Cantidad']
        
        fig3 = px.bar(top_grupos, x='Cantidad', y='Grupo', orientation='h',
                     title="Top 10 Grupos más Activos",
                     color='Cantidad', color_continuous_scale='Viridis')
        fig3.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Evolución temporal de los top 5 grupos
        top5_grupos = top_grupos['Grupo'].head(5).tolist()
        df_top5 = df_integrado[df_integrado['grupo'].isin(top5_grupos)]
        
        if 'año' in df_top5.columns:
            df_top5_anual = df_top5.groupby(['año', 'grupo']).size().reset_index(name='cantidad')
            
            fig4 = px.line(df_top5_anual, x='año', y='cantidad', color='grupo',
                         title="Evolución anual de los 5 grupos más activos",
                         labels={'año': 'Año', 'cantidad': 'Cantidad', 'grupo': 'Grupo'})
            fig4.update_layout(template='plotly_white')
            st.plotly_chart(fig4, use_container_width=True)
    
    # Análisis geográfico (si hay datos de país)
    if 'pais' in df_integrado.columns:
        st.subheader("Análisis Geográfico")
        
        # Top 10 países
        top_paises = df_integrado['pais'].value_counts().head(10).reset_index()
        top_paises.columns = ['País', 'Cantidad']
        
        fig5 = px.bar(top_paises, x='Cantidad', y='País', orientation='h',
                     title="Top 10 Países más Afectados",
                     color='Cantidad', color_continuous_scale='Viridis')
        fig5.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig5, use_container_width=True)
        
        # Mapa de calor geográfico
        df_geo = df_integrado.copy()
        
        # Geolocalizar países
        rows = []
        for pais, cnt in df_geo['pais'].value_counts().items():
            lat, lon = _geolocate(pais)
            if lat is not None:
                rows.append({"pais": pais, "count": cnt, "lat": lat, "lon": lon})
        
        df_map = pd.DataFrame(rows)
        
        if not df_map.empty:
            # Calcular radios para visualización
            max_count = df_map['count'].max()
            df_map['radius'] = df_map['count']/max_count*50000 + 20000
            
            # Crear mapa
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[200, 30, 0, 160],
                pickable=True,
                tooltip={"html": "<b>{pais}</b>: {count} registros"}
            )
            
            center = [df_map['lat'].mean(), df_map['lon'].mean()]
            view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=1.5)
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view,
                map_style="mapbox://styles/mapbox/dark-v10"
            )
            
            st.pydeck_chart(deck, use_container_width=True)
    
    # Análisis sectorial (si hay datos de sector)
    if 'sector' in df_integrado.columns:
        st.subheader("Análisis por Sector")
        
        # Distribución por sector
        sector_counts = df_integrado['sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Cantidad']
        
        # Mostrar solo los top 15 sectores para mejor visualización
        if len(sector_counts) > 15:
            otros_count = sector_counts.iloc[15:]['Cantidad'].sum()
            sector_counts = sector_counts.iloc[:15]
            sector_counts = pd.concat([
                sector_counts, 
                pd.DataFrame([{'Sector': 'Otros', 'Cantidad': otros_count}])
            ], ignore_index=True)
        
        fig6 = px.pie(sector_counts, values='Cantidad', names='Sector',
                     title="Distribución por Sector",
                     hole=0.4)
        fig6.update_layout(template='plotly_white')
        st.plotly_chart(fig6, use_container_width=True)
        
        # Evolución temporal por sector (top 5)
        if 'año' in df_integrado.columns:
            top5_sectores = sector_counts['Sector'].head(5).tolist()
            df_top5_sectores = df_integrado[df_integrado['sector'].isin(top5_sectores)]
            
            df_sectores_anual = df_top5_sectores.groupby(['año', 'sector']).size().reset_index(name='cantidad')
            
            fig7 = px.line(df_sectores_anual, x='año', y='cantidad', color='sector',
                         title="Evolución anual de los 5 sectores más afectados",
                         labels={'año': 'Año', 'cantidad': 'Cantidad', 'sector': 'Sector'})
            fig7.update_layout(template='plotly_white')
            st.plotly_chart(fig7, use_container_width=True)
    
    # Análisis de correlación entre grupos y sectores
    if {'grupo', 'sector'}.issubset(df_integrado.columns):
        st.subheader("Correlación entre Grupos y Sectores")
        
        # Crear matriz de grupos vs sectores
        grupo_sector = pd.crosstab(df_integrado['grupo'], df_integrado['sector'])
        
        # Seleccionar top 10 grupos y top 10 sectores para mejor visualización
        top10_grupos = df_integrado['grupo'].value_counts().head(10).index
        top10_sectores = df_integrado['sector'].value_counts().head(10).index
        
        matriz_reducida = grupo_sector.loc[top10_grupos.intersection(grupo_sector.index), top10_sectores.intersection(grupo_sector.columns)]
        
        if not matriz_reducida.empty:
            fig8 = px.imshow(matriz_reducida,
                           labels=dict(x="Sector", y="Grupo", color="Cantidad"),
                           title="Mapa de calor: Grupos vs Sectores",
                           color_continuous_scale='Viridis')
            fig8.update_layout(template='plotly_white')
            st.plotly_chart(fig8, use_container_width=True)
    

def show_group_analysis(df: pd.DataFrame) -> None:
    """
    Muestra análisis de los top 10 grupos y evolución temporal de los top 5.
    """
    if 'grupo' in df.columns:
        
        # Top 10 grupos
        top_grupos = (
            df['grupo']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_grupos.columns = ['Grupo', 'Cantidad']
        
        fig3 = px.bar(
            top_grupos,
            x='Cantidad',
            y='Grupo',
            orientation='h',
            # title="Top 10 Grupos más Activos"
            color='Cantidad',
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(
            template='plotly_white',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Evolución temporal de los top 5 grupos
        top5_grupos = top_grupos['Grupo'].head(5).tolist()
        df_top5 = df[df['grupo'].isin(top5_grupos)].copy()
        
        if 'fecha' in df_top5.columns:
            # Agregar columna de mes-año para la evolución
            df_top5['mes_ano'] = (
                pd.to_datetime(df_top5['fecha'], errors='coerce')
                .dt.to_period('M')
                .dt.to_timestamp()
            )
            
            evo = (
                df_top5
                .groupby(['mes_ano', 'grupo'])
                .size()
                .reset_index(name='Cantidad')
            )
            fig_line = px.line(
                evo,
                x='mes_ano',
                y='Cantidad',
                color='grupo',
                markers=True
                # title="Evolución Temporal de los Top 5 Grupos"
            )
            fig_line.update_layout(template='plotly_white')
            st.plotly_chart(fig_line, use_container_width=True)


def show_sector_analysis(df: pd.DataFrame) -> None:
    """
    Muestra análisis sectorial con distribución de sectores (top 15 + Otros).
    """
    if 'sector' in df.columns:
            
        # Distribución por sector
        sector_counts = (
            df['sector']
            .value_counts()
            .reset_index()
        )
        sector_counts.columns = ['Sector', 'Cantidad']
        
        # Mostrar solo los top 15 sectores para mejor visualización
        if len(sector_counts) > 15:
            otros_count = sector_counts.iloc[5:]['Cantidad'].sum()
            sector_counts = sector_counts.iloc[:5]
            sector_counts = pd.concat([
                sector_counts, 
                pd.DataFrame([{'Sector': 'Otros', 'Cantidad': otros_count}])
            ], ignore_index=True)
        
        fig6 = px.pie(
            sector_counts,
            values='Cantidad',
            names='Sector',
            # title="Distribución por Sector"
            hole=0.4
        )
        fig6.update_layout(template='plotly_white')
        st.plotly_chart(fig6, use_container_width=True)


def show_group_analysis(df: pd.DataFrame) -> None:
    """
    Muestra análisis de los top 10 grupos y evolución temporal de los top 5.
    """
    if 'grupo' in df.columns:
        
        # Top 10 grupos
        top_grupos = (
            df['grupo']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_grupos.columns = ['Grupo', 'Cantidad']
        
        fig3 = px.bar(
            top_grupos,
            x='Cantidad',
            y='Grupo',
            orientation='h',
            # title="Top 10 Grupos más Activos"
            color='Cantidad',
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(
            template='plotly_white',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Evolución temporal de los top 5 grupos
        top5_grupos = top_grupos['Grupo'].head(5).tolist()
        df_top5 = df[df['grupo'].isin(top5_grupos)].copy()
        
        if 'fecha' in df_top5.columns:
            # Agregar columna de mes-año para la evolución
            df_top5['mes_ano'] = (
                pd.to_datetime(df_top5['fecha'], errors='coerce')
                .dt.to_period('M')
                .dt.to_timestamp()
            )
            
            evo = (
                df_top5
                .groupby(['mes_ano', 'grupo'])
                .size()
                .reset_index(name='Cantidad')
            )
            fig_line = px.line(
                evo,
                x='mes_ano',
                y='Cantidad',
                color='grupo',
                markers=True
                # title="Evolución Temporal de los Top 5 Grupos"
            )
            fig_line.update_layout(template='plotly_white')
            st.plotly_chart(fig_line, use_container_width=True)


def plot_paises_mas_afectados(df: pd.DataFrame) -> None:
    """
    Muestra análisis de los top 10 países y evolución temporal de los top 5.
    """
    if 'pais' in df.columns:
        
        # Top 10 paises
        top_paises = (
            df['pais']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_paises.columns = ['pais', 'Cantidad']
        
        fig3 = px.bar(
            top_paises,
            x='Cantidad',
            y='pais',
            orientation='h',
            # title="Top 10 Países más Afectados"
            color='Cantidad',
            color_continuous_scale='Viridis'
        )
        fig3.update_layout(
            template='plotly_white',
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Evolución temporal de los top 5 países
        top5_paises = top_paises['pais'].head(5).tolist()
        df_top5 = df[df['pais'].isin(top5_paises)].copy()
        
        if 'fecha' in df_top5.columns:
            # Agregar columna de mes-año para la evolución
            df_top5['mes_ano'] = (
                pd.to_datetime(df_top5['fecha'], errors='coerce')
                .dt.to_period('M')
                .dt.to_timestamp()
            )
            
            evo = (
                df_top5
                .groupby(['mes_ano', 'pais'])
                .size()
                .reset_index(name='Cantidad')
            )
            fig_line = px.line(
                evo,
                x='mes_ano',
                y='Cantidad',
                color='pais',
                markers=True
                # title="Evolución Temporal de los Top 5 Países"
            )
            fig_line.update_layout(template='plotly_white')
            st.plotly_chart(fig_line, use_container_width=True)
