TravelIQ — Tourism Experience Analytics

Classification, Prediction, and Recommendation System built as a Labmentix portfolio project using Python, Streamlit, Pandas, Plotly, Scikit-learn, SQLite, and Joblib.

TravelIQ is an end-to-end tourism analytics platform for exploring tourism patterns, predicting attraction ratings, classifying likely visit modes, and recommending attractions.

Project Status

Complete — integrated, tested, and ready for deployment.

Included

Data cleaning and preprocessing

Exploratory data analysis and interactive visualizations

SQL-based tourism analysis

Rating prediction

Visit mode classification

Personalized attraction recommendations

Memory Passport traveler history

Premium dark travel-tech Streamlit interface

Dataset

49,208 tourism transactions

33,530 travelers

30 attractions in the transaction catalogue

5 visit modes

153 traveler-origin countries

2013–2022 historical period

4.16 / 5 average rating

The raw source data is intentionally excluded from the repository. Cleaned and processed datasets required by the application are included.

Machine Learning

Rating Prediction — Regression

Selected model: Gradient Boosting Regressor

Metric

Test Result

R²

0.0724

MAE

0.7752

MSE

0.9459

RMSE

0.9725

Visit Mode Classification

Selected model: Random Forest Classifier

Classes: Business, Couples, Family, Friends, Solo

Metric

Test Result

Accuracy

0.4250

Macro Precision

0.3134

Macro Recall

0.3522

Macro F1

0.2811

Attraction Recommendation

A deterministic content/profile-based recommender combines traveler attraction-type preferences, attraction quality, popularity, and visited-attraction exclusion. Cold-start recommendations are supported for new or unknown travelers.

Metric

Top-5

Top-10

Precision

0.1651

0.0988

Recall

0.6723

0.8062

MAP

0.4198

0.4401

The recommendation system was evaluated with a time-based train/test cutoff and checked for leakage, deterministic output, real attraction IDs, and visited-attraction exclusion.

Streamlit Pages

Home — tourism overview, KPIs, destinations, trending attractions, and world explorer

Tourism Analytics — visit modes, ratings, countries, attractions, yearly trends, and seasonality

Rating Predictor — estimates an expected attraction rating

Visit Mode Predictor — predicts Business, Couples, Family, Friends, or Solo

Attraction Recommendations — personalized or cold-start attraction suggestions

Memory Passport — traveler-specific history and memory stamps

About Project — objective, workflow, and technology stack

Tech Stack

Python · Streamlit · Pandas · Plotly · Scikit-learn · SQLite · Joblib

Project Structure

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
├── requirements.txt
└── README.md

Run Locally

pip install -r requirements.txt
streamlit run app.py

Data and Leakage Controls

Model training uses time-aware historical features. User and attraction statistics are calculated only from information available before the relevant visit period, with cold-start fallbacks when prior history is unavailable. The classification inference pipeline excludes the target rating.

Documentation

The docs/ directory contains the data dictionary, cleaning decisions, EDA report and artifacts, SQL analysis, model evaluation, recommendation evaluation, leakage checks, and reproducibility checks.

Deployment

Platform: Streamlit Community Cloud
Live application: To be added after deployment

Repository

https://github.com/Kusuma-Ramesh/TravelIQ-Tourism-Experience-Analytics

Project Goal

TravelIQ brings analytics, machine learning, recommendation, and interactive visualization together into one end-to-end tourism experience platform — helping users discover smarter and travel better.
