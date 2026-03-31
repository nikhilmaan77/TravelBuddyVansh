# Travel Buddy - Final Analytics Dashboard

## Project Overview
Travel Buddy is a trust-first solo travel companion platform designed to validate a new business idea through synthetic data, descriptive analytics, machine learning, and business interpretation.

The dashboard demonstrates an end-to-end analytics pipeline for a startup-style business validation project. It covers synthetic data generation, data cleaning and transformation, descriptive analytics, classification model comparison, clustering, and regression analysis.

## Business Problem
Solo travelers often struggle to find safe, trustworthy, and compatible journey companions. Existing travel or social platforms usually focus on discovery and swiping rather than verified compatibility and trust.

## Proposed Solution
Travel Buddy is a verified travel companion matching platform that uses:
- LinkedIn verification
- Passport verification
- trust-based matching logic
- profile completeness
- route compatibility
- satisfaction prediction

The platform follows a single-match logic rather than endless swiping, making the journey-planning experience more focused and safer.

## Assignment Coverage
This dashboard is designed to align with the academic project requirements:

1. Synthetic data generation for business validation  
2. Data cleaning and transformation  
3. Descriptive analytics / EDA with graphs and logical insights  
4. Advanced analytics including:
   - classification algorithms with evaluation
   - clustering with interpretation
   - Linear, Ridge, and Lasso regression  
5. Report-ready and presentation-ready summary sections

## Dashboard Sections
The Streamlit app includes the following sections:

1. **Executive Summary**  
   High-level KPIs and business validation overview.

2. **Data Cleaning & Transformation**  
   Shows preprocessing steps, data types, missing value handling, and transformed features.

3. **EDA / Descriptive Analytics**  
   Includes age distribution, satisfaction distribution, route analysis, transport analysis, and correlation matrix.

4. **Classification Models**  
   Compares multiple classifiers using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

   Models included:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - KNN
   - Naive Bayes
   - SVM
   - Gradient Boosting
   - XGBoost (if available)

5. **Clustering Analysis**  
   Uses K-Means clustering with:
   - elbow method
   - silhouette score
   - cluster profiling
   - business interpretation of traveler personas

6. **Regression Analysis**  
   Compares:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression

   Evaluation metrics:
   - R²
   - RMSE
   - MAE

7. **Report Summary**  
   A ready-to-use text summary for the final written report.

8. **Presentation Summary**  
   A ready-to-use slide outline for final presentation preparation.

## Synthetic Dataset
The application generates an embedded synthetic dataset and does not require an external CSV file.

Dataset characteristics include:
- 2,000 synthetic users
- verified user signals
- travel routes
- transport mode
- travel class
- trust score
- route compatibility
- satisfaction
- journey companion success

This dataset is used to validate the business idea in the absence of real startup transaction data.

## Key Variables
Some important fields used in analysis include:
- `LinkedIn_Verified`
- `Passport_Verified`
- `Profile_Completeness`
- `Trust_Score`
- `Route_Compatibility`
- `Journey_Companion_Found`
- `Satisfaction`
- `Trips_Per_Year`
- `Response_Time_Minutes`
- `Transport_Mode`
- `Travel_Class`

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Deployment Notes
The project is compatible with Streamlit Community Cloud or similar Python deployment environments.

## Academic Use
This dashboard is structured to support:
- report screenshots
- EDA discussion
- machine learning comparison tables
- clustering interpretation
- regression analysis
- presentation slide preparation

## Expected Outcome
The final dashboard supports the business case that trust, verification, route compatibility, and complete profiles improve successful companion matching and overall travel satisfaction.