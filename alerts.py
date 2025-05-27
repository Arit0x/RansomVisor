import streamlit as st
import feedparser
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from googletrans import Translator
from bs4 import BeautifulSoup
from datetime import datetime
from utils import (
    carga_datos_locales_recentcyberattacks,
    carga_datos_locales_recentvictims,
    ISO_TO_PAIS
)

# —————————————————————————————————————————————————————————————
# 0) Constantes de feeds
# —————————————————————————————————————————————————————————————
ALERTS_FEED             = "https://www.cisa.gov/uscert/ncas/alerts.xml"
BULLETINS_FEED          = "https://www.cisa.gov/uscert/ncas/bulletins.xml"
RANSOMWARE_FEED         = "https://www.ransomware.live/rss.xml"
INCIBE_AVISOS_FEED      = "https://www.incibe.es/incibe-cert/alerta-temprana/avisos/feed"
INCIBE_AVISOS_SCI_FEED  = "https://www.incibe.es/incibe-cert/alerta-temprana/avisos-sci/feed"
CERT_EU_FEED            = "https://cert.europa.eu/publications/security-advisories-rss"
NCSC_UK_REPORTS_FEED    = "https://www.ncsc.gov.uk/api/1/services/v1/report-rss-feed.xml"
NCSC_UK_NEWS_FEED       = "https://www.ncsc.gov.uk/api/1/services/v1/news-rss-feed.xml"
CNCS_PT_FEED            = "https://www.cncs.gov.pt/docs/noticias/feed-rss/index.xml"

MAX_CHARS_SNIPPET = 300     # caracteres máximos para traducir de cada contenido
PER_PAGE          = 5       # paginación: ítems por página

# —————————————————————————————————————————————————————————————
# 1) Cachés y utilidades
# —————————————————————————————————————————————————————————————
@st.cache_data(ttl=600)
def get_feed_entries(feed_url: str) -> list:
    """Obtiene y cachea las entradas de un RSS (10 min TTL)."""
    return feedparser.parse(feed_url).entries

@st.cache_resource
def get_translator():
    return Translator()

@st.cache_data(ttl=600)
def get_requests_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5,
                    status_forcelist=[500,502,503,504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({"Accept-Encoding":"gzip"})
    return sess

@st.cache_data(ttl=3600)
def fetch_conditional(url: str, etag: str=None, lastmod: str=None):
    """Descarga con ETag/If-Modified-Since. Devuelve (text, etag, lastmod) o (None, etag, lastmod)."""
    headers = {}
    if etag:    headers["If-None-Match"] = etag
    if lastmod: headers["If-Modified-Since"] = lastmod
    resp = get_requests_session().get(url, headers=headers, timeout=10)
    if resp.status_code == 304:
        return None, etag, lastmod
    resp.raise_for_status()
    return resp.text, resp.headers.get("ETag"), resp.headers.get("Last-Modified")

@lru_cache(maxsize=2048)
def translate_text(text: str, src="auto", dest="es") -> str:
    """
    LRU‐cache para traducciones repetidas.
    Captura errores de conexión y, si ocurre cualquier excepción,
    devuelve el texto original para no romper la UI.
    """
    try:
        return get_translator().translate(text, src=src, dest=dest).text
    except Exception:
        return text


def clean_html(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")

# —————————————————————————————————————————————————————————————
# 2) Prefetch en background — versión corregida
# —————————————————————————————————————————————————————————————
def prefetch_feeds():
    """Descarga todos los feeds oficiales en paralelo y cachea etags."""
    state = st.session_state

    # Si ya hemos prefetchado, salimos
    if "feed_data" in state:
        return

    # Inicializamos
    state.feed_data = {}
    state.feed_meta = {}

    feeds = [
        ALERTS_FEED, BULLETINS_FEED,
        INCIBE_AVISOS_FEED, INCIBE_AVISOS_SCI_FEED,
        CERT_EU_FEED, NCSC_UK_REPORTS_FEED,
        NCSC_UK_NEWS_FEED, CNCS_PT_FEED
    ]
    executor = ThreadPoolExecutor(max_workers=6)
    futures = {
        executor.submit(fetch_conditional, url,
                        state.feed_meta.get(url, {}).get("etag"),
                        state.feed_meta.get(url, {}).get("lastmod")): url
        for url in feeds
    }

    for fut in as_completed(futures):
        url = futures[fut]
        try:
            text, etag, lastmod = fut.result()
            if text is not None:
                entries = feedparser.parse(text).entries
                state.feed_data[url] = entries
                state.feed_meta[url] = {"etag": etag, "lastmod": lastmod}
            else:
                # No cambió, pero si no existía antes, inicializamos a lista vacía
                state.feed_data.setdefault(url, [])
        except:
            state.feed_data[url] = []
    executor.shutdown(wait=False)

# Llamamos a prefetch cuando arranca la app (o en show_official_alerts)
# prefetch_feeds()


# —————————————————————————————————————————————————————————————
# 3) Pipeline genérico de parse + snippet + traducción + paginación
# —————————————————————————————————————————————————————————————
def _parse_and_display(entries, page=0, key_prefix="default"):
    """Muestra una lista de entry‐like objetos con traducción completa bajo demanda."""
    start = page * PER_PAGE
    end   = start + PER_PAGE
    subset = entries[start:end]
    if not subset:
        st.warning("⚠️ No hay más entradas.")
        return False

    for idx, entry in enumerate(subset):
        # Campos originales
        title_orig = getattr(entry, "title", "🔔 Sin título")
        published_orig = getattr(entry, "published", "")
        country_orig = getattr(entry, "pais", "")
        clean_sum  = clean_html(
            getattr(entry, "summary", "") or
            getattr(entry, "description", "") or
            (entry.content[0]["value"] if getattr(entry, "content", []) else "")
        )
        snippet_orig = (
            clean_sum[:MAX_CHARS_SNIPPET].rsplit(" ",1)[0] + "…"
            if len(clean_sum)>MAX_CHARS_SNIPPET else clean_sum
        )

        # Render del original
        st.markdown(f"### 🔔 {title_orig}")
        st.markdown(f"📆 *{published_orig}*")
        if country_orig:
            st.markdown(f"🌍 **País:** {country_orig}")
        st.write(snippet_orig)

        # Checkbox para traducción completa
        chk_key = f"{key_prefix}-{page}-{idx}-translate-all"
        if st.checkbox("🈯 Traducir todo al español", key=chk_key):
            # Traduce cada parte (título, snippet, fecha y país se mantienen como strings)
            title_es = translate_text(title_orig)
            snippet_es = translate_text(snippet_orig)
            # Fecha: la dejamos igual o podrías formatearla si lo deseas
            published_es = translate_text(published_orig)
            country_es = translate_text(country_orig) if country_orig else ""

            # Mostrar traducción
            st.markdown(f"### 🔔 {title_es}")
            st.markdown(f"📆 *{published_es}*")
            if country_es:
                st.markdown(f"🌍 **País:** {country_es}")
            st.write(snippet_es)

        # Link y separador
        link = getattr(entry, "link", "#")
        if link and link != "#":
            st.markdown(f"[🔗 Leer más…]({link})")
        st.markdown("---")

    # Paginación con keys únicas
    cols = st.columns(3)
    prev_key = f"prev-{key_prefix}-{page}"
    next_key = f"next-{key_prefix}-{page}"
    if page > 0 and cols[0].button("« Anterior", key=prev_key):
        st.session_state.page = page - 1
        return True
    if len(entries) > end and cols[2].button("Siguiente »", key=next_key):
        st.session_state.page = page + 1
        return True

    return False




# —————————————————————————————————————————————————————————————
# 4.1) show_official_alerts — versión corregida
# —————————————————————————————————————————————————————————————
def show_official_alerts(limit_per_feed: int = 5):
    """Todas las alertas oficiales traducidas y paginadas, con fallback en caso de feed vacío."""
    state = st.session_state

    # Aseguramos que feed_data y feed_meta existen y han sido prefetchados
    if "feed_data" not in state:
        state.feed_data = {}
        state.feed_meta = {}
        prefetch_feeds()

    if "page" not in state:
        state.page = 0

    labels = [
        "US - US-CERT Alerts","US - US-CERT",
        "ES - INCIBE Alerts","ES - INCIBE Alerts SCI",
        "EU - CERT-EU Alerts","GB - NCSC Alerts",
        "GB - NCSC","PT - CNCS",
    ]
    feeds = [
        ALERTS_FEED, BULLETINS_FEED,
        INCIBE_AVISOS_FEED, INCIBE_AVISOS_SCI_FEED,
        CERT_EU_FEED, NCSC_UK_REPORTS_FEED,
        NCSC_UK_NEWS_FEED, CNCS_PT_FEED,
    ]

    tabs = st.tabs(labels)

    for tab_container, url in zip(tabs, feeds):
        with tab_container:
            # 1) Intentamos con la caché paralela
            entries = state.feed_data.get(url, [])
            # 2) Si está vacío (p.ej. CERT-EU), usamos feedparser directo
            if not entries:
                entries = get_feed_entries(url)
            # 3) Aplicamos el límite
            entries = entries[:limit_per_feed]

            # 4) Mostramos con paginación/traducción
            if _parse_and_display(entries, page=state.page, key_prefix=url):
                st.experimental_rerun()



# —————— 4.2 Ransomware.live feed with country mapping ——————
@st.cache_data(ttl=600)
def fetch_ransomware_feed(limit: int = 5) -> list:
    """Obtiene y cachea las últimas `limit` entradas del feed de ransomware.live, incluyendo mapeo de país."""
    entries = get_feed_entries(RANSOMWARE_FEED)[:limit]
    for entry in entries:
        # default
        entry.pais = ""
        # si vienen tags con country code, lo mapeamos
        if hasattr(entry, "tags"):
            for tag in entry.tags:
                code = tag.term.strip().upper()
                if code in ISO_TO_PAIS:
                    entry.pais = ISO_TO_PAIS[code]
                    break
    return entries

def show_ransomware_feed(limit: int = 5):
    # ...
    state = st.session_state
    state.page = 0
    entries = fetch_ransomware_feed(limit)
    if _parse_and_display(entries, page=state.page, key_prefix="ransomware"):
        st.experimental_rerun()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Fetch de Coveware News
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_coveware_news(limit: int = 5) -> list:
    sess = get_requests_session()
    url  = "https://www.coveware.com/ransomware-quarterly-reports"
    try:
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    items = soup.select("div.summary-item")[:limit]
    out   = []

    for itm in items:
        a = itm.select_one("div.summary-thumbnail-outer-container a")
        if not a:
            continue

        title_en = a.get("data-title", "").strip() or "Sin título"
        href     = a.get("href", "").strip()
        link     = href if href.startswith("http") else f"https://www.coveware.com{href}"

        # Extraemos todos los <p> dentro de summary-excerpt o summary-excerpt-only
        excerpt_div = itm.select_one("div.summary-excerpt") \
                   or itm.select_one("div.summary-excerpt-only")
        paras = []
        if excerpt_div:
            for p in excerpt_div.find_all("p"):
                text = p.get_text(" ", strip=True)
                if text:
                    paras.append(text)
        # Si por alguna razón no hay <p>, fallback a todo el texto bruto
        if not paras:
            raw = excerpt_div.get_text(" ", strip=True) if excerpt_div else ""
            paras = [raw] if raw else []

        time_tag = itm.select_one("time.summary-metadata-item--date")
        if time_tag and time_tag.has_attr("datetime"):
            published = time_tag["datetime"]
        elif time_tag:
            published = time_tag.get_text(strip=True)
        else:
            published = ""

        out.append(feedparser.FeedParserDict(
            title     = title_en,
            link      = link,
            published = published,
            paras     = paras,   # lista de párrafos en inglés
            description="",
            content   = []
        ))

    return out

# ──────────────────────────────────────────────────────────────────────────────
# 3) Mostrar Coveware: snippet + traducción con fallback
# ──────────────────────────────────────────────────────────────────────────────
def show_coveware_news(limit: int = 5):
    st.subheader("📰 Últimos Informes de Coveware")

    # Inicializar paginación
    if "coveware_page" not in st.session_state:
        st.session_state.coveware_page = 0
    page = st.session_state.coveware_page

    news = fetch_coveware_news(limit)
    if not news:
        st.warning("No se encontraron artículos.")
        return

    start = page * PER_PAGE
    end   = start + PER_PAGE

    for idx, art in enumerate(news[start:end]):
        title_en     = art["title"]
        published_en = art["published"]
        paras_en     = art["paras"]

        # 1) snippet rápido del primer párrafo
        first_para = paras_en[0] if paras_en else ""
        snippet_en = (
            first_para[:MAX_CHARS_SNIPPET].rsplit(" ", 1)[0] + "…"
            if len(first_para) > MAX_CHARS_SNIPPET else first_para
        )

        # 2) Mostrar original (en inglés)
        st.markdown(f"### 🔔 {title_en}")
        st.markdown(f"📆 *{published_en}*")
        st.write(snippet_en)

        # 3) Checkbox para traducir todo
        chk_key = f"coveware-trans-p{page}-i{idx}"
        if st.checkbox("🈯 Traducir al español", key=chk_key):
            # traducir título y fecha
            title_es     = translate_text(title_en) or title_en
            published_es = translate_text(published_en, src="auto") or published_en

            st.markdown(f"### 🔔 {title_es}")
            st.markdown(f"📆 *{published_es}*")

            # traducir cada párrafo; si falla, uso el párrafo original
            for p_en in paras_en:
                p_es = translate_text(p_en) or p_en
                st.write(p_es)

        # 4) Enlace a informe completo
        st.markdown(f"[🔗 Leer informe completo]({art['link']})")
        st.markdown("---")

    # Botones de paginación
    cols = st.columns([1,1,1])
    if page > 0 and cols[0].button("« Anterior", key="coveware-prev"):
        st.session_state.coveware_page -= 1
        st.experimental_rerun()
    if end < len(news) and cols[2].button("Siguiente »", key="coveware-next"):
        st.session_state.coveware_page += 1
        st.experimental_rerun()


# ─── ATAQUES RECIENTES EN PRENSA ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_recent_attacks(limit: int = 20) -> pd.DataFrame:
    """
    Carga recentcyberattacks.json de disco y devuelve sólo los últimos 'limit'
    incidentes, ordenados por fecha.
    """
    df = carga_datos_locales_recentcyberattacks()
    if df.empty:
        return df
    # Ordenamos por 'fecha' y tomamos los primeros 'limit'
    return df.sort_values("fecha", ascending=False).head(limit)


def show_recent_attacks(limit: int = 10):
    """📰 Últimos Ataques en Prensa (traducido)"""
    st.subheader("📰 Últimos Ataques en Prensa (traducido)")

    df = fetch_recent_attacks(limit)
    if df.empty:
        st.warning("No hay ataques recientes.")
        return

    # Métrica de fecha más reciente
    más_reciente = df["fecha"].iloc[0].date().isoformat()
    st.metric("Fecha más reciente", más_reciente)

    # Preparamos las entradas para el parser genérico
    entries = []
    for _, r in df.iterrows():
        published = r.fecha.isoformat()
        entries.append(feedparser.FeedParserDict(
            title       = r.titulo,
            link        = r.url,
            published   = published,
            summary     = r.resumen,
            description = "",
            content     = []
        ))

    # Paginación / despliegue
    st.session_state.page = 0
    if _parse_and_display(entries, page=st.session_state.page, key_prefix="recent-attacks"):
        st.experimental_rerun()


# ─── VÍCTIMAS RECIENTES ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_recent_victims(limit: int = 100) -> pd.DataFrame:
    """
    Carga recentvictims.json de disco, renombra columnas y devuelve
    sólo las últimas 'limit' víctimas, ordenadas por fecha.
    """
    df = carga_datos_locales_recentvictims()
    if df.empty:
        return df

    # Ya tenemos 'fecha','grupo','victima','pais','actividad','captura'
    df = df.sort_values("fecha", ascending=False).head(limit)
    return df


def show_recent_victims(limit: int = 10):
    """📢 Últimas Víctimas Detectadas (sin llamadas automáticas a la API)."""
    st.subheader("📢 Últimas Víctimas Detectadas")

    df = fetch_recent_victims(limit)
    if df.empty:
        st.warning("⚠️ No se encontraron víctimas en el snapshot local.")
        return

    # Métrica de fecha más reciente
    fecha_max = df["fecha"].dt.date.max().isoformat()
    st.metric("Fecha más reciente", fecha_max)

    # Listado de víctimas
    for _, row in df.iterrows():
        fecha     = row.fecha.date().isoformat()
        grupo     = row.grupo or "Desconocido"
        victima   = row.victima or "Desconocido"
        pais      = row.pais or "—"
        actividad = row.actividad or ""
        captura   = row.captura or ""

        line = f"**La víctima es {victima}** atacada por {grupo} en **{pais}** el {fecha}"
        if actividad.strip():
            line += f" — _{actividad}_"
        st.markdown(f"- {line}")

        if captura:
            try:
                st.image(captura, width=200, caption="Captura")
            except:
                st.markdown(f"[Ver captura]({captura})")

        st.markdown("---")