"""
Reusable glass-card style components.

Every card in the app should be built from these helpers so the
glassmorphism look stays consistent without duplicating CSS/HTML
across pages.
"""

import streamlit as st

ICON_CLASS_MAP = {
    "violet": "tiq-icon-violet",
    "sky": "tiq-icon-sky",
    "emerald": "tiq-icon-emerald",
    "orange": "tiq-icon-orange",
    "coral": "tiq-icon-coral",
}

GRADIENT_MAP = {
    "violet": "var(--gradient-primary)",
    "warm": "var(--gradient-warm)",
    "emerald": "var(--gradient-emerald)",
}


def kpi_card(icon: str, label: str, value: str, delta: str | None = None, color: str = "violet") -> str:
    """Return HTML for a single KPI card. Render with st.markdown(..., unsafe_allow_html=True)."""
    icon_class = ICON_CLASS_MAP.get(color, "tiq-icon-violet")
    delta_html = f'<div class="tiq-kpi-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="tiq-kpi">
        <div class="tiq-kpi-icon {icon_class}">{icon}</div>
        <div class="tiq-kpi-label">{label}</div>
        <div class="tiq-kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def render_kpi_row(kpis: list[dict]):
    """kpis: list of dicts with keys icon, label, value, delta (optional), color (optional)."""
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.markdown(
                kpi_card(
                    icon=kpi.get("icon", "📊"),
                    label=kpi.get("label", ""),
                    value=kpi.get("value", "—"),
                    delta=kpi.get("delta"),
                    color=kpi.get("color", "violet"),
                ),
                unsafe_allow_html=True,
            )


def glass_card_open(extra_class: str = ""):
    st.markdown(f'<div class="tiq-card {extra_class}">', unsafe_allow_html=True)


def glass_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def section_title(icon: str, title: str):
    st.markdown(
        f'<div class="tiq-section-title"><span>{icon}</span><span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="tiq-empty-state">
            <div class="tiq-empty-icon">{icon}</div>
            <div class="tiq-empty-title">{title}</div>
            <div class="tiq-empty-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_placeholder(label: str, unit: str, message: str):
    """A polished 'model not connected yet' state that still looks like a
    real result card — big dash value, unit, and a short explanation."""
    st.markdown(
        f"""
        <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
            <div class="tiq-eyebrow" style="margin-bottom:0.6rem;">{label}</div>
            <div style="font-size:2.6rem; font-weight:800; color:var(--text-muted); line-height:1;">— {unit}</div>
            <div class="tiq-empty-title" style="margin-top:1rem;">Model not connected yet</div>
            <div class="tiq-empty-sub">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    subtitle_html = f'<div class="tiq-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-bottom:1.4rem;">
            <div class="tiq-eyebrow">{eyebrow}</div>
            <div class="tiq-hero-title" style="font-size:1.9rem;">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, new: bool = False) -> str:
    cls = "tiq-badge tiq-badge-new" if new else "tiq-badge"
    return f'<span class="{cls}">{text}</span>'
