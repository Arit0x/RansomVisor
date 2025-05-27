from utils import load_css
import alerts
import streamlit as st

def alertas_app():

    # Inyectar CSS global
    load_css()

    # ─── SIDEBAR ──────────────────────────────────────────────────────────────────
    # display_logo()

    # ─── CONTENIDO PRINCIPAL ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "Oficiales",
        "Ransomware.live",
        "Coveware News",
    ])

    with tabs[0]:
        st.subheader("Alertas Oficiales")
        alerts.show_official_alerts(limit_per_feed=5)

    with tabs[1]:
        sub_tabs = st.tabs([
            "Víctimas Recientes",
            "Ataques Recientes",
            "Feeds"
        ])
        
        with sub_tabs[0]:
            alerts.show_recent_victims(limit=5)
        with sub_tabs[1]:
            alerts.show_recent_attacks(limit=5)
        with sub_tabs[2]:
            alerts.show_ransomware_feed(limit=5)

    with tabs[2]:
        st.subheader("Coveware News")
        alerts.show_coveware_news(limit=5)

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