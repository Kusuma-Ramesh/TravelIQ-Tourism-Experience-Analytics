# TravelIQ — Tourism Experience Analytics

Classification, Prediction, and Recommendation System — a Streamlit
frontend for a Labmentix portfolio project.

## Status: Part 2 — Premium UI, Animations, and Memory Passport

Part 1 shipped the architecture and navigation. This part adds the full
glassmorphism polish pass, 200–400ms hover/transition treatment
throughout, the signature scrapbook-style **Memory Passport**, a
sidebar passport preview card, polished predictor/analytics empty
states, and a small hidden easter egg. **Still no data cleaning, EDA,
or ML** — all values remain explicit demo data.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
tourism_project/
├── app.py                  # entrypoint: page config, CSS, navigation/routing
├── pages/
│   ├── home.py              # hero + KPI overview + preview panels
│   ├── analytics.py         # Tourism Analytics (empty states)
│   ├── rating_predictor.py  # Rating Predictor (empty states)
│   ├── visit_mode.py        # Visit Mode Predictor (empty states)
│   ├── recommendations.py   # Attraction Recommendations (empty states)
│   ├── passport.py          # Memory Passport (scrapbook stamps, demo data)
│   └── about.py             # About Project
├── components/
│   ├── navigation.py         # single source of truth for the nav structure
│   ├── sidebar.py            # custom glass sidebar (brand, nav, traveler footer)
│   ├── cards.py               # KPI cards, glass cards, page headers, empty states
│   ├── passport.py            # stamp + progress-bar visuals
│   └── charts.py               # themed plotly chart helpers
├── styles/
│   └── main.css                 # glassmorphism dark travel-tech theme
├── data/                          # (empty — real dataset lands in a later part)
├── models/                        # (empty — trained models land in a later part)
└── assets/                        # (empty — static assets go here)
```

## Notes for the next parts

- `data/`, `models/` are placeholders — nothing reads from them yet.
- KPI values in `pages/home.py` and stamps in `pages/passport.py` are
  hardcoded demo data, clearly commented as such — swap for real
  dataframe calculations once the pipeline exists.
- Predictor pages disable their submit buttons and show
  "model not connected yet" placeholders rather than fabricating
  results (see `components/cards.py::prediction_placeholder`).
- Sidebar passport counts (`components/sidebar.py::render_footer`) are
  demo values — wire them to real visit history alongside the full
  passport page.
- The compass easter egg on the Memory Passport page lives entirely in
  `pages/passport.py` via `st.session_state["compass_clicks"]` — no
  custom JS component involved.
- Google Fonts (`Caveat`, `Space Grotesk`) are pulled in via `@import`
  at the top of `styles/main.css` for the passport's handwritten look;
  swap for self-hosted fonts if the deployment target has no internet
  access to fonts.googleapis.com.
