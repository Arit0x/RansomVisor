import requests
import pandas as pd
import json
from pathlib import Path
import streamlit as st
from datetime import datetime

# ─── CONFIG: DIRECTORIOS DE DATOS ───────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# ─── 1. SESIÓN HTTP REUTILIZABLE ────────────────────────────────────────────
@st.cache_resource
def get_http_session() -> requests.Session:
    """Devuelve una sesión persistente para todas las llamadas HTTP."""
    return requests.Session()

# ─── 2. LÓGICA DE ACTUALIZACIÓN ────────────────────────────────────────────────
def run_update_flow():
    """Ejecuta los 3 pasos de actualización y marca el flag en session_state."""
    total_steps = 3
    step = 0

    placeholder = st.container()
    progress_bar = placeholder.progress(0)

    # Paso 1
    old_ataques = load_snapshot("recentcyberattacks.json")
    ataques = actualizar_recentcyberattacks()
    delta_ataques = len(ataques) - len(old_ataques)
    step += 1; progress_bar.progress(int(step/total_steps*100))

    # Paso 2
    old_victims = load_snapshot("recentvictims.json")
    victimas = actualizar_recentvictims()
    delta_victims = len(victimas) - len(old_victims)
    step += 1; progress_bar.progress(int(step/total_steps*100))

    # Paso 3
    old_ano = load_snapshot("flattened_ransomware_year.json")
    victimas_ano = actualizar_snapshot_victimas_por_ano()
    delta_ano = len(victimas_ano) - len(old_ano)
    step += 1; progress_bar.progress(int(step/total_steps*100))

    placeholder.empty()

    st.session_state.update_result = {
        "ataques": ataques, "delta_ataques": delta_ataques,
        "victimas": victimas, "delta_victims": delta_victims,
        "victimas_ano": victimas_ano, "delta_ano": delta_ano
    }
    st.session_state.show_summary = True


def show_update_summary():
    """Muestra el resumen de la actualización en la página."""
    res = st.session_state.update_result
    st.success("✅ Actualización completada")
    c1, c2, c3 = st.columns(3)
    c1.metric("📰 Ataques recientes", len(res["ataques"]), delta=res["delta_ataques"])
    c2.metric("📢 Víctimas recientes", len(res["victimas"]), delta=res["delta_victims"])
    c3.metric("🗓️ Víctimas por año", len(res["victimas_ano"]), delta=res["delta_ano"])
    if st.button("Cerrar resumen", key="home_close_summary"):
        st.session_state.show_summary = False
        st.session_state.update_result = None

# ─── 3. CONSULTA API CON CACHE ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def consulta_api(endpoint: str, params: dict | None = None) -> dict:
    """
    Llama a la API de ransomware.live y devuelve JSON.
    Cachea el resultado durante 1 hora.
    """
    url = f"https://api.ransomware.live/v2/{endpoint}"
    resp = get_http_session().get(url, params=params, headers={"Accept": "application/json"})
    resp.raise_for_status()
    try:
        return resp.json() or {}
    except ValueError:
        return {}

# ─── 4. FUNCIONES GENÉRICAS DE SNAPSHOT ─────────────────────────────────────
def load_snapshot(filename: str) -> list[dict]:
    """Carga JSON local de data/filename."""
    path = DATA_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_snapshot(data: list[dict], filename: str) -> None:
    """Guarda data en JSON 'data/filename'."""
    path = DATA_DIR / filename
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fetch_incremental(
    endpoint: str,
    old_records: list[dict],
    date_field: str | None = 'date',
    params_extra: dict | None = None
) -> list[dict]:
    """
    Obtiene sólo los registros nuevos para endpoint según date_field,
    y desempaqueta payloads dict con clave 'data' en listas.
    """
    params = {}
    if date_field and old_records:
        df_old = pd.DataFrame(old_records)
        if date_field in df_old.columns:
            df_old[date_field] = pd.to_datetime(df_old[date_field], errors='coerce')
            last = df_old[date_field].max()
            if pd.notna(last):
                params['since'] = last.isoformat()
    if params_extra:
        params.update(params_extra)

    raw = consulta_api(endpoint, params) or []

    if isinstance(raw, dict):
        raw_list = raw.get('data') if isinstance(raw.get('data'), list) else []
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    if date_field and old_records and date_field in df_old.columns:
        df_new = pd.DataFrame(raw_list)
        if date_field in df_new.columns:
            df_new[date_field] = pd.to_datetime(df_new[date_field], errors='coerce')
            df_new = df_new[df_new[date_field] > last]
            raw_list = df_new.to_dict('records')

    return raw_list


def update_snapshot_generic(
    endpoint: str,
    filename: str,
    key_field: str = 'id',
    date_field: str | None = 'date',
    params_extra: dict | None = None
) -> list[dict]:
    """Actualiza snapshot: merge incremental + deduplicación."""
    old = load_snapshot(filename)
    inc = fetch_incremental(endpoint, old, date_field, params_extra)
    if inc:
        combined = old + inc
        df = pd.DataFrame(combined)
        if key_field and key_field in df.columns:
            df = df.drop_duplicates(subset=key_field, keep='last')
        records = df.to_dict('records')
        save_snapshot(records, filename)
        return records
    return old

# ─── 5. WRAPPERS PARA VÍCTIMAS POR AÑO ────────────────────────────────────────
def carga_snapshot_victimas_por_ano() -> list[dict]:
    return load_snapshot("flattened_ransomware_year.json")


def guarda_snapshot_victimas_por_ano(data: list[dict]) -> None:
    save_snapshot(data, "flattened_ransomware_year.json")


from datetime import datetime

def actualizar_snapshot_victimas_por_ano(start_year: int = 2021) -> list[dict]:
    """
    Si no hay datos locales, descarga todos los años desde start_year hasta el actual;
    en ejecuciones posteriores, solo trae los años nuevos (last_year+1 ... current_year),
    acumulando sin deduplicar dentro de cada año.
    """
    # 1) Cargo lo viejo y determino el último año procesado (o None si no hay)
    old = carga_snapshot_victimas_por_ano()
    años_procesados = [r.get("year") for r in old if isinstance(r.get("year"), int)]
    last_year = max(años_procesados) if años_procesados else None

    # 2) Año actual y rango a consultar
    current_year = datetime.utcnow().year
    if last_year is None:
        años_a_consultar = range(start_year, current_year + 1)
    else:
        años_a_consultar = range(last_year + 1, current_year + 1)

    # 3) Arranco all_records con todo lo viejo
    all_records = list(old)

    # 4) Recorro cada año nuevo y acumulo sin deduplicar
    for yr in años_a_consultar:
        raw = consulta_api(f"victims/{yr}") or {}
        # desempaco payload si viene en dict
        lista = raw.get("victims") if isinstance(raw, dict) and isinstance(raw.get("victims"), list) else raw
        if not isinstance(lista, list):
            continue

        for v in lista:
            entry = v.copy() if isinstance(v, dict) else {"value": v}
            entry["year"] = yr
            all_records.append(entry)

    # 5) Serializo, guardo y devuelvo
    guarda_snapshot_victimas_por_ano(all_records)
    return all_records


# ─── 7. WRAPPERS PARA ATAQUES RECIENTES EN PRENSA ────────────────────────────
def carga_snapshot_recentcyberattacks() -> list[dict]:
    return load_snapshot("recentcyberattacks.json")


def actualizar_recentcyberattacks() -> list[dict]:
    return update_snapshot_generic(
        endpoint='recentcyberattacks',
        filename='recentcyberattacks.json',
        key_field='url',
        date_field='date'
    )

# ─── 8. WRAPPERS PARA VÍCTIMAS RECIENTES ────────────────────────────────────
def carga_snapshot_recentvictims() -> list[dict]:
    return load_snapshot("recentvictims.json")


def actualizar_recentvictims() -> list[dict]:
    return update_snapshot_generic(
        endpoint='recentvictims',
        filename='recentvictims.json',
        key_field='victim',
        date_field='attackdate'
    )

# ─── 9. NORMALIZACIÓN DE PAÍS, GRUPO Y SECTOR ────────────────────────────────
ISO_TO_PAIS = {
    'AD': 'Andorra', 'AE': 'Emiratos Árabes Unidos', 'AF': 'Afganistán', 'AG': 'Antigua y Barbuda',
    'AI': 'Anguila', 'AL': 'Albania', 'AM': 'Armenia', 'AO': 'Angola', 'AR': 'Argentina', 'AT': 'Austria',
    'AU': 'Australia', 'AW': 'Aruba', 'AZ': 'Azerbaián', 'BA': 'Bosnia y Herzegovina', 'BB': 'Barbados',
    'BD': 'Bangladés', 'BF': 'Burkina Faso', 'BG': 'Bulgaria', 'BH': 'Baréin', 'BI': 'Burundi', 'BM': 'Bermudas',
    'BN': 'Brunéi', 'BO': 'Bolivia', 'BR': 'Brasil', 'BS': 'Bahamas', 'BW': 'Botsuana', 'BY': 'Bielorrusia',
    'BZ': 'Belice', 'CA': 'Canadá', 'CG': 'Congo', 'CH': 'Suiza', 'CI': 'Costa de Marfil', 'CL': 'Chile',
    'CM': 'Camerún', 'CN': 'China', 'CO': 'Colombia', 'CP': 'Islas Cook', 'CR': 'Costa Rica', 'CU': 'Cuba',
    'CW': 'Curazao', 'CY': 'Chipre', 'CZ': 'República Checa', 'DE': 'Alemania', 'DJ': 'Yibuti', 'DK': 'Dinamarca',
    'DM': 'Dominica', 'DN': 'Dinamarca', 'DO': 'República Dominicana', 'DZ': 'Argelia', 'EC': 'Ecuador', 'EE': 'Estonia',
    'EG': 'Egipto', 'ER': 'Eritrea', 'ES': 'España', 'ET': 'Etiopía', 'EU': 'Unión Europea', 'FI': 'Finlandia',
    'FJ': 'Fiyi', 'FO': 'Islas Feroe', 'FR': 'Francia', 'GA': 'Gabón', 'GB': 'Reino Unido', 'GE': 'Georgia', 'GH': 'Ghana',
    'GI': 'Gibraltar', 'GL': 'Groenlandia', 'GM': 'Gambia', 'GR': 'Grecia', 'GT': 'Guatemala', 'GU': 'Países Bajos',
    'GY': 'Guyana', 'HK': 'Hong Kong', 'HN': 'Honduras', 'HR': 'Croacia', 'HT': 'Haití', 'HU': 'Hungría', 'ID': 'Indonesia',
    'IE': 'Irlanda', 'IL': 'Israel', 'IN': 'India', 'IQ': 'Irak', 'IR': 'Irán', 'IS': 'Islandia', 'IT': 'Italia',
    'JA': 'Japón', 'JE': 'Jersey', 'JM': 'Jamaica', 'JO': 'Jordania', 'JP': 'Japón', 'KE': 'Kenia', 'KG': 'Kirguistán',
    'KH': 'Camboya', 'KI': 'Kiribati', 'KR': 'Corea del Sur', 'KW': 'Kuwait', 'KY': 'Islas Caimán', 'KZ': 'Kazajistán',
    'LA': 'Laos', 'LB': 'Líbano', 'LK': 'Sri Lanka', 'LS': 'Lesoto', 'LT': 'Lituania', 'LU': 'Luxemburgo', 'LV': 'Letonia',
    'LY': 'Libia', 'MA': 'Marruecos', 'MC': 'Mónaco', 'MD': 'Moldavia', 'ME': 'Montenegro', 'MG': 'Madagascar',
    'MK': 'Macedonia del Norte', 'ML': 'Malí', 'MN': 'Mongolia', 'MO': 'Macao', 'MP': 'Islas Marianas del Norte',
    'MR': 'Mauritania', 'MU': 'Mauricio', 'MW': 'Malaui', 'MX': 'México', 'MY': 'Malasia', 'MZ': 'Mozambique', 'NA': 'Namibia',
    'NC': 'Nueva Caledonia', 'ND': 'Níger', 'NE': 'Níger', 'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Países Bajos', 'NO': 'Noruega',
    'NP': 'Nepal', 'NZ': 'Nueva Zelanda', 'OM': 'Omán', 'PA': 'Panamá', 'PE': 'Perú', 'PF': 'Polinesia Francesa',
    'PG': 'Papúa Nueva Guinea', 'PH': 'Filipinas', 'PK': 'Pakistán', 'PL': 'Polonia', 'PN': 'Islas Pitcairn', 'PO': 'Polonia',
    'PT': 'Portugal', 'PW': 'Palaos', 'PY': 'Paraguay', 'QA': 'Catar', 'QC': 'Quebec', 'RE': 'Reunión', 'RO': 'Rumanía',
    'RS': 'Serbia', 'RU': 'Rusia', 'RW': 'Ruanda', 'SA': 'Arabia Saudita', 'SB': 'Islas Salomón', 'SC': 'Seychelles', 'SD': 'Sudán',
    'SE': 'Suecia', 'SG': 'Singapur', 'SK': 'Eslovaquia', 'SL': 'Sierra Leona', 'SN': 'Senegal', 'SO': 'Somalia', 'SP': 'España',
    'SR': 'Surinam', 'ST': 'Santo Tomé y Príncipe', 'SV': 'El Salvador', 'SW': 'Suecia', 'SY': 'Siria', 'TC': 'Islas Turcas y Caicos',
    'TD': 'Chad', 'TH': 'Tailandia', 'TJ': 'Tayikistán', 'TL': 'Timor Oriental', 'TN': 'Túnez', 'TR': 'Turquía',
    'TT': 'Trinidad y Tobago', 'TU': 'Túnez', 'TV': 'Tuvalu', 'TZ': 'Tanzania', 'UA': 'Ucrania', 'UG': 'Uganda', 'UK': 'Reino Unido',
    'US': 'Estados Unidos', 'UY': 'Uruguay', 'UZ': 'Uzbekistán', 'VC': 'San Vicente y las Granadinas', 'VG': 'Islas Vírgenes Británicas',
    'VI': 'Islas Vírgenes EE.UU.', 'VN': 'Vietnam', 'WS': 'Samoa', 'ZA': 'Sudáfrica', 'ZM': 'Zambia', 'ZW': 'Zimbabue',
    'BE': 'Bélgica', 'MM': 'Myanmar', 'MT': 'Malta', 'PR': 'Puerto Rico', 'TW': 'Taiwán', 'VE': 'Venezuela',
    'ca': 'Canadá', 'cl': 'Chile', 'lk': 'Sri Lanka', 'tz': 'Tanzania', 'us': 'Estados Unidos'
}

GRUPO_SYNONYMS = {
    False: None, 'Lockbit':'LockBit', 'Lock Bit':'LockBit',
    'Revil':'REvil', 'Conti':'Conti', 'Maze':'Maze', 'Ryuk':'Ryuk'
}

ES_TO_SECTOR = {
    'Servicios empresariales':'Business Services', 'Tecnología':'Technology', 'Manufactura':'Manufacturing',
    'Salud':'Healthcare', 'Transporte y logística':'Transportation/Logistics', 'Finanzas':'Financial',
    'Gobierno':'Government', 'Agricultura y alimentación':'Agriculture and Food Production', 'Energía':'Energy',
    'Educación':'Education', 'Hostelería y turismo':'Hospitality and Tourism', 'Servicios al consumidor':'Consumer Services',
    'Sector público':'Public Sector', 'Servicios financieros':'Financial Services','Instalaciones gubernamentales':'Government Facilities',
    'Construcción':'Construction', 'TIC':'Information Technology', 'Salud pública':'Healthcare and Public Health',
    'Manufactura crítica':'Critical Manufacturing', 'Telecomunicaciones':'Telecommunication','Sistemas de transporte':'Transportation Systems',
    'Centros educativos':'Education Facilities','Comunicación':'Communication','Instalaciones comerciales':'Commercial Facilities',
    'Alimentos y agricultura':'Food and Agriculture','Servicios de emergencia':'Emergency Services','Comercio minorista':'Retail',
    'Mayorista y minorista':'Wholesale & Retail','Ingeniería':'Engineering','Química':'Chemical','Publicidad, marketing y RRPP':'Advertising, Marketing & PR',
    'Energía y suministros':'Energy & Utilities','Internet y telecomunicaciones':'Internet & Telecom Services','Base industrial de defensa':'Defense Industrial Base',
    'Transporte':'Transportation','Servicios legales':'Law Firms & Legal Services','Automotriz':'Automotive','ONGs':'Community, Social Services & Non-Profit Organisations',
    'Aeroespacial':'Aerospace','Alimentos y bebidas':'Food & Beverages','Otros':'Others','Radiodifusión':'Broadcasting',
    'Inmobiliario':'Real Estate','Tratamiento de agua':'Water and Wastewater Systems','Nuclear':'Nuclear Reactors, Materials, and Waste',
    'Fabricación TI':'IT Manufacturing','Envíos y logística':'Shipping & Logistics','Legal':'Legal'
}

SECTOR_MAP_ESP = {
    'Information Technology': 'Tecnología de la información',
    'Financial': 'Finanzas',
    'Business Services': 'Servicios empresariales',
    '': 'Desconocido',
    'Manufacturing': 'Manufactura',
    'Hospitality and Tourism': 'Hostelería y turismo',
    'Government': 'Gobierno',
    'Not Found': 'Desconocido',
    'Healthcare': 'Salud',
    'Engineering': 'Ingeniería',
    'Government Facilities': 'Instalaciones gubernamentales',
    'Healthcare Services': 'Servicios de salud',
    'Wholesale & Retail': 'Comercio mayorista y minorista',
    'Law Firms & Legal Services': 'Servicios legales y bufetes de abogados',
    'Advertising, Marketing & Public Relations': 'Publicidad, marketing y relaciones públicas',
    'Aerospace': 'Aeroespacial',
    'Transportation/Logistics': 'Transporte y logística',
    'Communication': 'Comunicación',
    'Energy': 'Energía',
    'Telecommunication': 'Telecomunicaciones',
    'Agriculture': 'Agricultura',
    'Healthcare and Public Health': 'Salud pública y asistencia sanitaria',
    'Transportation Systems': 'Sistemas de transporte',
    'Technology': 'Tecnología',
    'Critical Manufacturing': 'Manufactura crítica',
    'Education Facilities': 'Instalaciones educativas',
    'Food and Agriculture': 'Alimentación y agricultura',
    'Commercial Facilities': 'Instalaciones comerciales',
    'Nuclear Reactors, Materials, and Waste': 'Reactores nucleares, materiales y residuos',
    'Chemical': 'Industria química',
    'Agriculture and Food Production': 'Agricultura y producción alimentaria',
    'Others': 'Otros',
    'Internet & Telecommunication Services': 'Servicios de internet y telecomunicaciones',
    'Education': 'Educación',
    'Energy & Utilities': 'Energía y servicios públicos',
    'Transportation': 'Transporte',
    'Automotive': 'Automoción',
    'Community, Social Services & Non-Profit Organisations': 'Servicios sociales y organizaciones sin ánimo de lucro',
    'Broadcasting': 'Radiodifusión',
    'Financial Services': 'Servicios financieros',
    'Defense Industrial Base': 'Industria de defensa',
    'Food & Beverages': 'Alimentación y bebidas',
    'Shipping & Logistics': 'Envíos y logística',
    'IT Manufacturing': 'Fabricación de TI',
    'Public Sector': 'Sector público',
    'Construction': 'Construcción',
    'Retail': 'Comercio minorista',
    'Consumer Services': 'Servicios al consumidor',
    'Real Estate': 'Bienes raíces',
    'Legal': 'Legal'
}
SECTOR_MAP_ESP.update(ES_TO_SECTOR)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'country' in df.columns:
        pais_series = df['country']
    elif 'pais' in df.columns:
        pais_series = df['pais']
    else:
        pais_series = pd.Series([pd.NA] * len(df), index=df.index)

    df['pais'] = (
        pais_series
        .fillna('UNKNOWN')
        .astype(str)
        .str.strip()
        .replace(ISO_TO_PAIS)
        .fillna('Desconocido')
    )

    if 'claim_gang' in df.columns:
        grupo_series = df['claim_gang']
    elif 'grupo' in df.columns:
        grupo_series = df['grupo']
    else:
        grupo_series = pd.Series([pd.NA] * len(df), index=df.index)

    df['grupo'] = (
        grupo_series
        .fillna(False)
        .replace(GRUPO_SYNONYMS)
        .fillna('Desconocido')
        .astype(str)
        .str.strip()
        .str.title()
    )

    if 'sector' in df.columns:
        df['sector'] = df['sector'].map(SECTOR_MAP_ESP).fillna(df['sector'])

    return df

# ─── 10. LOAD LOCAL SNAPSHOTS (SIN LLAMAR A LA API) ───────────────────────
@st.cache_data(ttl=3600)
def carga_datos_locales_alertas() -> pd.DataFrame:
    records = load_snapshot("alerts.json")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.rename(columns={'timestamp':'fecha_alerta'})
    df['fecha_alerta'] = pd.to_datetime(df['fecha_alerta'], errors='coerce')
    return normalize_data(df)

@st.cache_data(ttl=3600)
def carga_datos_locales_recentcyberattacks() -> pd.DataFrame:
    records = load_snapshot("recentcyberattacks.json")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)[[
        "claim_gang","country","date","victim","title","summary","url"
    ]]
    df = df.rename(columns={
        "claim_gang":"grupo","country":"pais","date":"fecha",
        "victim":"victima","title":"titulo","summary":"resumen"
    })
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    return normalize_data(df)

@st.cache_data(ttl=3600)
def carga_datos_locales_recentvictims() -> pd.DataFrame:
    records = load_snapshot("recentvictims.json")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "attackdate":"fecha","group":"grupo","victim":"victima",
        "country":"pais","activity":"actividad","screenshot":"captura"
    })
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    return normalize_data(df)

@st.cache_data(ttl=3600)
def carga_datos_victimas_por_ano() -> pd.DataFrame:
    records = carga_snapshot_victimas_por_ano()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    cols = ['activity', 'attackdate', 'group', 'victim', 'country']
    df = df.loc[:, [c for c in cols if c in df.columns]]
    df = df.rename(columns={
        'activity':    'sector',
        'attackdate':  'fecha',
        'group':       'grupo',
        'victim':      'victima',
        'country':     'pais'
    })
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['pais'] = (
        df['pais']
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .fillna('Desconocido')
    )
    df['sector'] = (
        df['sector']
        .astype(str)
        .str.strip()
        .replace(r'(?i)^(?:|not\s*found)$', pd.NA, regex=True)
        .fillna('Desconocido')
    )
    return normalize_data(df)

# ─── XX. CSS LEGADO ─────────────────────────────────────────
@st.cache_resource
def load_css(path: str = 'assets/style.css') -> None:
    file = Path(path)
    if file.exists():
        st.markdown(f"<style>{file.read_text()}</style>", unsafe_allow_html=True)
