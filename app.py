import streamlit as st
from utils import load_css

# 1) Configuración global
st.set_page_config(
    page_title="RansomVisor",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Inicialización global de session_state ────────────────
if "update_result" not in st.session_state:
    st.session_state.update_result = None
if "show_summary" not in st.session_state:
    st.session_state.show_summary = False

# 2) Carga estilos globales 
load_css()

# 3) Importa páginas
from pages.home import home_app
from pages.tendencias import tendencias_app
from pages.geografia import geografia_app
from pages.alertas import alertas_app
from pages.modelado_modular import modelado_modular_app
from pages.sectores import sectores_app

# 4) Define menú de navegación
home_page      = st.Page(home_app,      title="Visión General", icon="📊")
tendencias_page= st.Page(tendencias_app,title="Tendencias",     icon="📈")
geografia_page = st.Page(geografia_app, title="Geografía",      icon="🌎")
alertas_page   = st.Page(alertas_app,   title="Alertas",        icon="🟠")
modelado_page  = st.Page(modelado_modular_app, title="Modelado",icon="🤖")
sectores_page  = st.Page(sectores_app,  title="Sectores",      icon="🏭")

# 5) Ejecuta el router
pg = st.navigation([
    home_page,
    tendencias_page,
    geografia_page,
    alertas_page,
    modelado_page,
    sectores_page
])
pg.run()
