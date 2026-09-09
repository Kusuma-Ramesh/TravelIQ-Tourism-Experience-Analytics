"""
Single source of truth for the app's navigation structure.

Both app.py (to build the st.Page list) and components/sidebar.py
(to render the custom nav) import from here, so adding a page never
means updating two places and risking a dead link.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    key: str            # unique id, also used as the pages/ filename stem
    label: str           # shown in the sidebar
    icon: str            # emoji icon
    file: str            # path relative to project root
    default: bool = False


NAV_ITEMS: list[NavItem] = [
    NavItem(key="home", label="Home", icon="🏠", file="pages/home.py", default=True),
    NavItem(key="analytics", label="Tourism Analytics", icon="📊", file="pages/analytics.py"),
    NavItem(key="rating_predictor", label="Rating Predictor", icon="⭐", file="pages/rating_predictor.py"),
    NavItem(key="visit_mode", label="Visit Mode Predictor", icon="🧳", file="pages/visit_mode.py"),
    NavItem(key="recommendations", label="Attraction Recommendations", icon="🗺️", file="pages/recommendations.py"),
    NavItem(key="passport", label="Memory Passport", icon="🛂", file="pages/passport.py"),
    NavItem(key="about", label="About Project", icon="ℹ️", file="pages/about.py"),
]


def get_nav_item(key: str) -> NavItem | None:
    for item in NAV_ITEMS:
        if item.key == key:
            return item
    return None
