"""About Project — Labmentix project brief summary."""

import streamlit as st

from components.cards import page_header, section_title

page_header(
    eyebrow="ℹ️ About",
    title="About This Project",
    subtitle="Tourism Experience Analytics: Classification, Prediction, and Recommendation System.",
)

col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown(
        """<div class="tiq-card">
    <h3>🎯 Project Goal</h3>
    <p>
    TravelIQ is a tourism analytics and machine learning platform built
    around a real tourism dataset. It combines data analysis, prediction,
    classification, and personalized recommendation into a single
    interactive travel-tech application.
    </p>
    <ul>
    <li><b>Classification</b> — predicts a traveler's likely visit mode.</li>
    <li><b>Prediction</b> — estimates the rating a traveler is likely to give an attraction.</li>
    <li><b>Recommendation</b> — suggests attractions using traveler preferences and visit history.</li>
    </ul>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tiq-card">
            <h3>🧱 Project Components</h3>
            <ol>
                <li>Data cleaning & preprocessing</li>
                <li>Exploratory data analysis & visualization</li>
                <li>SQL-based tourism analysis</li>
                <li>Rating prediction model</li>
                <li>Visit mode classification model</li>
                <li>Personalized attraction recommendation engine</li>
                <li>Streamlit integration & interactive dashboard</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tiq-card">
            <h3>📦 Project Status</h3>
            <p>
            TravelIQ integrates the cleaned tourism dataset, exploratory
            analytics, trained machine learning models, recommendation engine,
            interactive visualizations, and Streamlit application into one
            end-to-end project.
            </p>
            <p>
            Built as part of the <b>Labmentix</b> program.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("🧱", "Project Workflow")
    st.markdown(
        """
        <div style="color:var(--text-secondary); font-size:0.9rem; line-height:1.8;">
        1. Data cleaning & preprocessing<br/>
        2. Exploratory data analysis & visualization<br/>
        3. SQL-based tourism analysis<br/>
        4. Rating prediction<br/>
        5. Visit mode classification<br/>
        6. Personalized attraction recommendation<br/>
        7. Streamlit application integration & validation
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("🛠️", "Tech Stack")
    for label in [
        "Streamlit",
        "Python",
        "Pandas",
        "Plotly",
        "Scikit-learn",
        "SQLite",
        "Joblib",
    ]:
        st.markdown(
            f'<span class="tiq-badge" style="margin:0.2rem 0.3rem 0.2rem 0;">{label}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("📦", "Source")
    st.markdown(
        """
        <div style="color:var(--text-secondary); font-size:0.88rem;">
        Built as part of the Labmentix program. The project integrates
the cleaned tourism dataset, analytics, SQL analysis, trained
machine learning models, recommendation engine, and interactive
Streamlit dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
