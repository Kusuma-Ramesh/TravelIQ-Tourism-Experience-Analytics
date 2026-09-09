# TravelIQ — Tourism Experience Analytics

> **Classification, Prediction, and Recommendation System**  
> An end-to-end tourism analytics and machine learning platform built as a Labmentix portfolio project.

TravelIQ combines tourism data analysis, machine learning, recommendation, SQL analytics, and an interactive Streamlit dashboard into one travel-tech application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Dataset Facts](#key-dataset-facts)
- [Key Features](#key-features)
- [Machine Learning](#machine-learning)
- [Streamlit Application](#streamlit-application)
- [Analytics and SQL](#analytics-and-sql)
- [Data Leakage and Validation](#data-leakage-and-validation)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Project Goal](#project-goal)

---

## Project Overview

TravelIQ is designed to answer three practical tourism questions:

| Capability | What it does |
|---|---|
| **Rating Prediction** | Estimates the rating a traveler is likely to give an attraction |
| **Visit Mode Classification** | Predicts Business, Couples, Family, Friends, or Solo |
| **Attraction Recommendation** | Suggests attractions using traveler preferences, historical quality, and popularity |

### Project Status

**Complete — integrated, tested, and ready for deployment.**

---

## Key Dataset Facts

| Metric | Value |
|---|---:|
| Tourism transactions | **49,208** |
| Travelers | **33,530** |
| Attractions | **30** |
| Visit modes | **5** |
| Traveler-origin countries | **153** |
| Historical period | **2013–2022** |
| Average rating | **4.16 / 5** |

The raw source dataset is intentionally excluded from the repository. Cleaned and processed datasets required by the application are included.

---

## Key Features

### Tourism Analytics

- Tourism transaction KPIs
- Visit-mode distribution
- Rating distribution
- Traveler-origin analysis
- Attraction popularity
- Yearly tourism trends
- Monthly seasonality
- Average rating trends
- Top- and bottom-rated attractions
- Interactive world explorer

### Prediction and Recommendation

- Rating prediction
- Visit mode prediction
- Personalized recommendations
- Cold-start recommendations
- Previously visited attraction exclusion

### Memory Passport

- Traveler-specific visit history
- Destinations explored
- Attractions explored
- Average rating
- Most common visit mode
- Travel-year summary
- Memory stamps

---

# Machine Learning

## 1. Rating Prediction — Regression

**Selected model:** Gradient Boosting Regressor

| Metric | Test Result |
|---|---:|
| **R²** | **0.0724** |
| **MAE** | **0.7752** |
| **MSE** | **0.9459** |
| **RMSE** | **0.9725** |

The training pipeline uses time-aware historical features to prevent future information from being used for earlier visits.

---

## 2. Visit Mode Classification

**Selected model:** Random Forest Classifier

| Class |
|---|
| Business |
| Couples |
| Family |
| Friends |
| Solo |

### Test Performance

| Metric | Test Result |
|---|---:|
| **Accuracy** | **0.4250** |
| **Macro Precision** | **0.3134** |
| **Macro Recall** | **0.3522** |
| **Macro F1** | **0.2811** |

The inference pipeline uses safe pre-visit features and excludes the target `Rating`.

---

## 3. Attraction Recommendation

TravelIQ uses a deterministic **content/profile-based recommendation approach**.

| Component | Weight |
|---|---:|
| Traveler attraction-type affinity | **60%** |
| Attraction quality | **25%** |
| Attraction popularity | **15%** |

### Recommendation Evaluation

| Metric | Top-5 | Top-10 |
|---|---:|---:|
| **Precision** | **0.1651** | **0.0988** |
| **Recall** | **0.6723** | **0.8062** |
| **MAP** | **0.4198** | **0.4401** |

Cold-start recommendations use attraction quality and popularity when traveler history is unavailable.

---

# Streamlit Application

| Page | Purpose |
|---|---|
| **Home** | Tourism overview, KPIs, destinations, trending attractions, and world explorer |
| **Tourism Analytics** | Interactive tourism statistics and visualizations |
| **Rating Predictor** | Estimate an expected attraction rating |
| **Visit Mode Predictor** | Predict Business, Couples, Family, Friends, or Solo |
| **Attraction Recommendations** | Personalized or cold-start attraction suggestions |
| **Memory Passport** | Traveler-specific history and memory stamps |
| **About Project** | Objective, workflow, and technology stack |

---

# Analytics and SQL

### Exploratory Data Analysis

The EDA covers traveler demographics, visit-mode patterns, attraction popularity, rating behavior, geographic distribution, yearly trends, monthly seasonality, and attraction comparisons.

### SQL Analysis

A SQLite database is included with structured tourism tables and reusable SQL queries for analytical questions.

---

# Data Leakage and Validation

Model training uses **time-aware historical features**. User and attraction statistics are calculated only from information available before the relevant visit period.

| Split | Historical period |
|---|---|
| Training | Jan 2013 – Jul 2017 |
| Validation | Aug 2017 – May 2018 |
| Test | Jun 2018 – Oct 2022 |

Cold-start fallbacks are used when prior history is unavailable. The project also includes leakage checks and reproducibility checks.

---

# Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core development and machine learning |
| **Streamlit** | Interactive web application |
| **Pandas** | Data processing and analysis |
| **Plotly** | Interactive visualizations |
| **Scikit-learn** | Machine learning and preprocessing |
| **SQLite** | SQL-based tourism analysis |
| **Joblib** | Model artifact storage and loading |

---

# Project Structure

```text
tourism_project/
├── app.py
├── pages/
├── components/
├── data/
│   ├── cleaned/
│   └── processed/
├── database/
├── docs/
├── models/
│   ├── regression/
│   ├── classification/
│   └── recommendation/
├── scripts/
├── styles/
│   └── main.css
├── requirements.txt
└── README.md
```

---

# Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# Documentation

| Location | Contents |
|---|---|
| `docs/data_dictionary.md` | Dataset fields and cleaning decisions |
| `docs/eda_report.md` | Exploratory analysis |
| `docs/eda/` | EDA statistics and supporting artifacts |
| `docs/model_evaluation_report.md` | Model evaluation |
| `database/` | SQL schema, queries, and database |

---

# Deployment

**Platform:** Streamlit Community Cloud

**Live application:** _To be added after deployment_

**GitHub:** https://github.com/Kusuma-Ramesh/TravelIQ-Tourism-Experience-Analytics

---

# Limitations

- The recommendation interaction catalogue contains **30 attractions**, while the broader source catalogue is larger.
- Visit-mode classification is affected by class imbalance, particularly for Business.
- Rating prediction has modest held-out R² and should be interpreted as an estimate.
- Predictions beyond the historical 2013–2022 period are extrapolations.

---

# Project Goal

TravelIQ brings **analytics, machine learning, recommendation, and interactive visualization** together into one end-to-end tourism experience platform.

> **Discover smarter. Travel better.**
