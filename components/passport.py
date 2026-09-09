"""
Reusable "Memory Passport" visuals — the signature scrapbook feature.

Pure presentation components; pages/passport.py and components/sidebar.py
supply the (currently demo) data. Each visited stamp gets a distinct
shape + color so the passport doesn't read as a repeated dashboard
widget — see styles/main.css for the shape/paper-theme rules.
"""

import streamlit as st

STAMP_COLORS = ["#2F6FB0", "#C23B3B", "#2E8B57", "#5B4CD6", "#B0552F"]
STAMP_SHAPES = ["circle", "scallop", "hex", "triangle", "diamond"]
TILTS = [-4, 3, -2, 5, -5, 2]


def progress_stat(label: str, current: int, total: int, color: str = "var(--accent-sky)"):
    pct = 0 if total == 0 else round((current / total) * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:0.9rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                <span style="color:var(--text-secondary); font-weight:600;">{label}</span>
                <span style="color:var(--text-muted);">{current} / {total}</span>
            </div>
            <div class="tiq-progress-track">
                <div class="tiq-progress-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stamp_html(entry: dict, index: int) -> str:
    tilt = TILTS[index % len(TILTS)]
    if entry["status"] == "visited":
        color = STAMP_COLORS[index % len(STAMP_COLORS)]
        shape = STAMP_SHAPES[index % len(STAMP_SHAPES)]
        html = f"""
        <div class="tiq-stamp-cell" style="--tilt:{tilt}deg;">
            <div style="position:relative; width:100%;">
                <div class="tiq-washi" style="--stamp-color:{color}; --tape-tilt:{tilt/2}deg;"></div>
                <div class="tiq-stamp-mark shape-{shape}" style="--stamp-color:{color};">
                    <div class="tiq-stamp-icon">🌍</div>
                    <div class="tiq-stamp-country">{entry['country']}</div>
                    <div class="tiq-stamp-city">{entry.get('city', '')}</div>
                    <div class="tiq-stamp-date">{entry.get('date', '')}</div>
                </div>
            </div>
            <div class="tiq-stamp-caption">{entry['country']}</div>
        </div>
        """
        return "\n".join(line.strip() for line in html.strip().splitlines())
    html = f"""
    <div class="tiq-stamp-cell" style="--tilt:{tilt}deg;">
        <div class="tiq-stamp-mark shape-square">
            <div class="tiq-stamp-locked-icon">🔒</div>
            <div class="tiq-stamp-locked-label">{entry['country']}</div>
            <div class="tiq-stamp-locked-tag">Not yet</div>
        </div>
        <div class="tiq-stamp-reveal">Maybe this will be<br/>your next adventure…</div>
    </div>
    """
    return "\n".join(line.strip() for line in html.strip().splitlines())


def stamp_grid(entries: list[dict]):
    """entries: list of {country, city, status, date}. Renders inside a paper-themed grid."""
    stamps_html = "\n\n".join(_stamp_html(entry, i) for i, entry in enumerate(entries))
    grid_html = f'<div class="tiq-stamp-grid">\n\n{stamps_html}\n\n</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def sidebar_passport_preview(countries: int, attractions: int, memories: int):
    """Compact 'Memory Passport' teaser card for the sidebar footer."""
    st.markdown(
        f"""
        <div class="tiq-sidebar-passport">
            <div class="tiq-sidebar-passport-label">🛂 Memory Passport</div>
            <div class="tiq-sidebar-passport-sub">Your journey so far</div>
            <div class="tiq-sidebar-passport-stats">
                <div>
                    <div class="tiq-sidebar-passport-stat-value">{countries}</div>
                    <div class="tiq-sidebar-passport-stat-label">Destinations</div>
                </div>
                <div>
                    <div class="tiq-sidebar-passport-stat-value">{attractions}</div>
                    <div class="tiq-sidebar-passport-stat-label">Attractions</div>
                </div>
                <div>
                    <div class="tiq-sidebar-passport-stat-value">{memories}</div>
                    <div class="tiq-sidebar-passport-stat-label">Memories</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="sidebar_passport_cta"):
        if st.button("Open Passport →", key="sidebar_passport_btn"):
            st.switch_page("pages/passport.py")
