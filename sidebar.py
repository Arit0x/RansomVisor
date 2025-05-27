import streamlit as st
import base64
from pathlib import Path
from datetime import date
import pandas as pd
from utils import load_css, ES_TO_SECTOR


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade filtros de Sector, Grupo, País y Rango de fechas al sidebar y devuelve df filtrado.
    """
    # ─── Filtrar por sector ────────────────────────────────────────────────────
    sector_es, sector_en = select_sector(key="sidebar_sector")
    if sector_es == "Todos (Sin Desconocidos)":
        df = df[df['sector'] != "Desconocido"]
    elif sector_en is not None:
        df = df[df['sector'] == sector_en]

    # ─── Filtrar por grupo ────────────────────────────────────────────────────
    opciones_grupo = ["Todos"] + sorted(df['grupo'].unique())
    grupo_sel = st.sidebar.selectbox(
        "👥 Grupo",
        opciones_grupo,
        index=0,
        key="sidebar_grupo"
    )
    if grupo_sel != "Todos":
        df = df[df['grupo'] == grupo_sel]

    # ─── Filtrar por país ─────────────────────────────────────────────────────
    opciones_pais = ["Todos", "Todos (Sin Desconocidos)"] + sorted(df['pais'].unique())
    pais_sel = st.sidebar.selectbox(
        "🌐 País",
        opciones_pais,
        index=0,
        key="sidebar_pais"
    )
    if pais_sel == "Todos (Sin Desconocidos)":
        df = df[df['pais'] != "Desconocido"]
    elif pais_sel != "Todos":
        df = df[df['pais'] == pais_sel]

    # ─── Filtrar por rango de fechas ───────────────────────────────────────────
    df = df.copy()
    df['fecha_only'] = pd.to_datetime(df['fecha'], errors='coerce').dt.normalize()

    min_ts = df['fecha_only'].min()
    max_ts = df['fecha_only'].max()
    min_date = min_ts.date() if pd.notna(min_ts) else date.today()
    max_date = max_ts.date() if pd.notna(max_ts) else date.today()

    fecha_sel = st.sidebar.date_input(
        "📅 Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="sidebar_fecha"
    )
    if isinstance(fecha_sel, tuple) and len(fecha_sel) == 2:
        start_date, end_date = fecha_sel
        mask = (
            (df['fecha_only'] >= pd.Timestamp(start_date)) &
            (df['fecha_only'] <= pd.Timestamp(end_date))
        )
        df = df[mask]

    df = df.drop(columns=['fecha_only'])

    # Línea divisoria en el sidebar
    st.sidebar.markdown("<hr style='border-top:1px solid #444;'/>", unsafe_allow_html=True)
    return df


def render_sidebar_controls(only_return_defaults=False):
    """
    Dibuja los controles de modelado y feeds, o solo devuelve valores predeterminados.
    
    Args:
        only_return_defaults (bool): Si es True, solo devuelve los valores predeterminados 
                                    sin crear elementos UI.
    
    Returns:
        tuple: (enable_advanced, include_events, detect_changepoints, 
                outlier_method, enable_regressors, dynamic_seasonality, show_metrics)
    """
    # Valores predeterminados (se usarán si la UI no los modifica)
    enable_advanced = True  # Siempre será True ya que no hay modelo básico
    include_events = True  # Activado por defecto para mejor precisión
    detect_changepoints = True  # Activado por defecto para mejor precisión
    outlier_method = "cap"  # Mejor opción general para outliers
    enable_regressors = True  # Activado por defecto para usar CVEs
    dynamic_seasonality = False  # Más avanzado, por defecto desactivado
    show_metrics = True  # Siempre mostrar métricas
    
    # Si solo queremos los valores predeterminados, devolvemos sin crear UI
    if only_return_defaults:
        return enable_advanced, include_events, detect_changepoints, outlier_method, enable_regressors, dynamic_seasonality, show_metrics
    
    # Selector de tipo de modelo (esta parte siempre se mostrará en el sidebar)
    st.sidebar.subheader("Tipo de Modelo 🔮")
    model_type = st.sidebar.radio(
        "Tipo de predicción",
        options=["Conteo de ataques por día", "Días entre ataques"],
        index=0,
        help="Selecciona el enfoque de modelado que mejor se ajuste a tus necesidades"
    )
    
    # Explicación del modelo seleccionado
    if model_type == "Conteo de ataques por día":
        st.sidebar.caption("📈 Predice cuántos ataques ocurrirán cada día")
    else:
        st.sidebar.caption("⏱️ Predice el tiempo que transcurrirá entre ataques consecutivos")
    
    # Solo mostramos controles avanzados si los demás controles están en el desplegable de la página principal
    # El resto de la configuración se ha movido a la página principal
    
    # Devolver valores configurados
    return enable_advanced, include_events, detect_changepoints, outlier_method, enable_regressors, dynamic_seasonality, show_metrics


def select_sector(key: str = "select_sector") -> tuple[str, str | None]:
    """
    Control de selección de sector en Español.
    Devuelve (sector_es, sector_en).
    """
    opciones = ["Todos", "Todos (Sin Desconocidos)"] + sorted(ES_TO_SECTOR.keys())
    sector_es = st.sidebar.selectbox(
        "🏭 Sector",
        opciones,
        index=0,
        key=key
    )
    if sector_es == "Todos (Sin Desconocidos)":
        sector_en = None
    elif sector_es == "Todos":
        sector_en = None
    else:
        sector_en = ES_TO_SECTOR.get(sector_es)
    return sector_es, sector_en
