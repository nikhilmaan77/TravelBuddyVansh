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
- Trust-based matching logic
- Profile completeness
- Route compatibility
- Satisfaction prediction

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

## Dashboard Sections
The Streamlit app includes the following sections:

1. **Profile Builder**  
   Shows profile completion patterns and match success impact.

2. **KPI Overview**  
   High-level KPIs and business validation overview.

3. **Global Routes**  
   Route-level success analysis across important travel corridors, including a world map of top 20 routes.

4. **Transport Analytics**  
   Transport mode and class performance insights.

5. **Demographics**  
   Age and trust-based segmentation of verified users.

6. **Match Engine**  
   Trust and matching performance analysis.

7. **Satisfaction**  
   Satisfaction outcomes based on match status and trust.

8. **Advanced Analytics**  
   Includes formal analytics modules for classification, clustering, and regression.

## Sidebar Features
The dashboard includes a demographic filter for **Age Group** in the left panel, allowing interactive exploration of all visualizations across selected age segments.

## Advanced Analytics

### Classification Models
The dashboard compares multiple classifiers using:
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

### Clustering Analysis
The dashboard uses K-Means clustering with:
- Elbow method
- Silhouette score
- Cluster profiling
- Business interpretation of traveler personas

### Regression Analysis
The dashboard compares:
- Linear Regression
- Ridge Regression
- Lasso Regression

Evaluation metrics:
- R²
- RMSE
- MAE

## Synthetic Dataset
The application generates an embedded synthetic dataset and does not require an external CSV file.

Dataset characteristics include:
- 2,400 synthetic users
- Verified user signals
- Travel routes
- Transport mode
- Travel class
- Trust score
- Route compatibility
- Satisfaction
- Journey companion success
- Trips per year
- Response time minutes
- Derived verification counts
- Age-group transformation
- Expanded city-level coordinates for route mapping

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
- `Transport_Mode`
- `Travel_Class`
- `Verified_Total`
- `Age_Group`
- `Trips_Per_Year`
- `Response_Time_Minutes`
- `City_From`
- `City_To`
- `Country_From`
- `Country_To`

## Route Map Coverage
The interactive global map includes major city coordinates and route visibility for key travel nodes such as:
- Mumbai
- Pune
- Bangalore
- Delhi
- Singapore
- Dubai
- Amsterdam
- London
- Berlin
- NYC
- San Francisco
- Paris

This ensures that major origin-destination routes are visible in the dashboard map and improves presentation quality for report screenshots.

## Sankey Improvement
The KPI Overview tab includes a Sankey diagram with updated distinct contrasting colors for:
- All users
- Single verified users
- Double verified users
- Matched users

This improves readability and makes user-conversion flow easier to interpret visually.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost (optional)

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Deployment Files
Recommended supporting files:

### `requirements.txt`
```txt
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
```

### `runtime.txt`
```txt
python-3.11
```

### `packages.txt`
```txt
build-essential
```

## Deployment Notes
The project is compatible with Streamlit Community Cloud or similar Python deployment environments.

## Academic Use
This dashboard is structured to support:
- Report screenshots
- EDA discussion
- Machine learning comparison tables
- Clustering interpretation
- Regression analysis
- Business validation storytelling

## Expected Outcome
The final dashboard supports the business case that trust, verification, route compatibility, and complete profiles improve successful companion matching and overall travel satisfaction.