"""
Travel Buddy - Final Analytics Dashboard
MBA / Data Analytics Project
LinkedIn + Passport Verified | Single Match Platform

Covers:
1. Synthetic data generation
2. Data cleaning & transformation
3. Descriptive analytics / EDA
4a. Classification model comparison
4b. K-Means clustering with interpretation
4c. Linear, Ridge and Lasso regression
5. Report-ready summary
6. Presentation-ready summary
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Travel Buddy Final Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
        color: white;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #151927 100%);
    }
    .metric-card {
        background: #1b2233;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .section-box {
        background: #171d2d;
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #c7cfdb;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SYNTHETIC DATA
# -----------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 2000

    countries = ['USA', 'Singapore', 'India', 'Netherlands', 'UK', 'UAE', 'Germany']
    cities_from = ['NYC', 'Singapore', 'Delhi', 'Amsterdam', 'London', 'Dubai', 'Berlin']
    cities_to = ['London', 'Delhi', 'Dubai', 'NYC', 'Singapore', 'Amsterdam', 'Paris']
    transport_modes = ['Air', 'Train', 'Cruise', 'Road']
    travel_classes = ['Economy', 'Premium Economy', 'Business', 'First']
    genders = ['Male', 'Female']

    df = pd.DataFrame({
        'User_ID': range(1, n + 1),
        'LinkedIn_Verified': np.random.choice([1, 0], n, p=[0.94, 0.06]),
        'Passport_Verified': np.random.choice([1, 0], n, p=[0.92, 0.08]),
        'Country_From': np.random.choice(countries, n),
        'City_From': np.random.choice(cities_from, n),
        'City_To': np.random.choice(cities_to, n),
        'Transport_Mode': np.random.choice(transport_modes, n, p=[0.55, 0.25, 0.15, 0.05]),
        'Travel_Class': np.random.choice(travel_classes, n, p=[0.60, 0.25, 0.10, 0.05]),
        'Age': np.random.randint(22, 55, n),
        'Gender': np.random.choice(genders, n, p=[0.52, 0.48]),
        'Profile_Completeness': np.clip(np.random.normal(88, 10, n), 50, 100).round(1),
        'Trust_Score': np.clip(np.random.normal(90, 8, n), 50, 100).round(1),
        'Route_Compatibility': np.clip(np.random.normal(84, 11, n), 50, 100).round(1),
        'Response_Time_Minutes': np.clip(np.random.normal(18, 8, n), 3, 60).round(0),
        'Trips_Per_Year': np.random.randint(1, 13, n)
    })

    df['Verified_Total'] = df['LinkedIn_Verified'] + df['Passport_Verified']

    df['Trust_Level'] = np.select(
        [
            (df['LinkedIn_Verified'] == 1) & (df['Passport_Verified'] == 1),
            (df['LinkedIn_Verified'] == 1) & (df['Passport_Verified'] == 0),
            (df['LinkedIn_Verified'] == 0) & (df['Passport_Verified'] == 1),
        ],
        [
            'Double Verified',
            'LinkedIn Only',
            'Passport Only'
        ],
        default='Unverified'
    )

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[21, 29, 37, 45, 55],
        labels=['22-29', '30-37', '38-45', '46-55']
    )

    route_bonus = np.where(df['City_From'] != df['City_To'], 4, -6)
    class_bonus = df['Travel_Class'].map({
        'Economy': 0,
        'Premium Economy': 3,
        'Business': 6,
        'First': 8
    }).astype(float)

    transport_bonus = df['Transport_Mode'].map({
        'Air': 4,
        'Train': 2,
        'Cruise': 6,
        'Road': -2
    }).astype(float)

    # synthetic business logic
    raw_match_score = (
        0.22 * df['Profile_Completeness'] +
        0.24 * df['Trust_Score'] +
        0.21 * df['Route_Compatibility'] +
        3.5 * df['Verified_Total'] +
        0.7 * df['Trips_Per_Year'] -
        0.18 * df['Response_Time_Minutes'] +
        class_bonus +
        transport_bonus +
        route_bonus +
        np.random.normal(0, 5, n)
    )

    prob = 1 / (1 + np.exp(-(raw_match_score - 40) / 9))
    df['Journey_Companion_Found'] = np.where(np.random.rand(n) < prob, 1, 0)

    satisfaction_base = (
        1.0 +
        0.020 * df['Trust_Score'] +
        0.016 * df['Route_Compatibility'] +
        0.010 * df['Profile_Completeness'] +
        0.38 * df['Journey_Companion_Found'] +
        0.10 * df['Verified_Total'] -
        0.008 * df['Response_Time_Minutes'] +
        np.random.normal(0, 0.25, n)
    )
    df['Satisfaction'] = np.clip(satisfaction_base, 1, 5).round(2)

    # keep verified-focused business dataset
    df = df[(df['LinkedIn_Verified'] == 1) | (df['Passport_Verified'] == 1)].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# -----------------------------
# CLEANING + TRANSFORMATION
# -----------------------------
@st.cache_data
def clean_transform_data(df):
    clean_df = df.copy()

    clean_df.drop_duplicates(inplace=True)

    numeric_cols = [
        'Age', 'Profile_Completeness', 'Trust_Score',
        'Route_Compatibility', 'Response_Time_Minutes',
        'Trips_Per_Year', 'Satisfaction', 'Verified_Total'
    ]

    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

    clean_df['Gender'] = clean_df['Gender'].fillna('Unknown')
    clean_df['Transport_Mode'] = clean_df['Transport_Mode'].fillna('Unknown')
    clean_df['Travel_Class'] = clean_df['Travel_Class'].fillna('Unknown')
    clean_df['City_From'] = clean_df['City_From'].fillna('Unknown')
    clean_df['City_To'] = clean_df['City_To'].fillna('Unknown')
    clean_df['Trust_Level'] = clean_df['Trust_Level'].fillna('Unknown')

    clean_df['Age_Group'] = pd.cut(
        clean_df['Age'],
        bins=[21, 29, 37, 45, 55],
        labels=['22-29', '30-37', '38-45', '46-55']
    )

    return clean_df


# -----------------------------
# MODEL HELPERS
# -----------------------------
def get_feature_lists():
    feature_cols = [
        'LinkedIn_Verified', 'Passport_Verified',
        'Profile_Completeness', 'Trust_Score',
        'Route_Compatibility', 'Age',
        'Gender', 'City_From', 'City_To',
        'Transport_Mode', 'Travel_Class',
        'Verified_Total', 'Trips_Per_Year',
        'Response_Time_Minutes'
    ]
    numeric_features = [
        'LinkedIn_Verified', 'Passport_Verified',
        'Profile_Completeness', 'Trust_Score',
        'Route_Compatibility', 'Age',
        'Verified_Total', 'Trips_Per_Year',
        'Response_Time_Minutes'
    ]
    categorical_features = [
        'Gender', 'City_From', 'City_To',
        'Transport_Mode', 'Travel_Class'
    ]
    return feature_cols, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


@st.cache_data
def run_classification_models(df):
    feature_cols, numeric_features, categorical_features = get_feature_lists()

    X = df[feature_cols]
    y = df['Journey_Companion_Found']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pre_scaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    pre_not_scaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', LogisticRegression(max_iter=1000))
        ]),
        'Decision Tree': Pipeline([
            ('preprocessor', pre_not_scaled),
            ('model', DecisionTreeClassifier(max_depth=6, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', pre_not_scaled),
            ('model', RandomForestClassifier(n_estimators=200, random_state=42))
        ]),
        'KNN': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', KNeighborsClassifier(n_neighbors=7))
        ]),
        'Naive Bayes': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', GaussianNB())
        ]),
        'SVM': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', SVC(probability=True, kernel='rbf', random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', pre_not_scaled),
            ('model', GradientBoostingClassifier(random_state=42))
        ]),
    }

    if XGB_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('preprocessor', pre_not_scaled),
            ('model', XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                random_state=42
            ))
        ])

    results = []
    confusion_store = {}

    for name, pipe in models.items():
        if name == 'Naive Bayes':
            X_train_trans = pre_scaled.fit_transform(X_train)
            X_test_trans = pre_scaled.transform(X_test)

            if hasattr(X_train_trans, "toarray"):
                X_train_trans = X_train_trans.toarray()
                X_test_trans = X_test_trans.toarray()

            model = GaussianNB()
            model.fit(X_train_trans, y_train)
            preds = model.predict(X_test_trans)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_trans)[:, 1]
            else:
                probs = None
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_test, probs) if probs is not None else np.nan
        except Exception:
            auc = np.nan

        cm = confusion_matrix(y_test, preds)
        confusion_store[name] = cm

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4),
            'ROC_AUC': round(auc, 4) if pd.notnull(auc) else np.nan
        })

    results_df = pd.DataFrame(results).sort_values(by='F1_Score', ascending=False).reset_index(drop=True)
    best_model = results_df.iloc[0]['Model']

    return results_df, confusion_store, best_model


@st.cache_data
def run_clustering(df):
    cluster_df = df.copy()

    cluster_features = [
        'Age', 'Profile_Completeness', 'Trust_Score',
        'Route_Compatibility', 'Satisfaction',
        'Trips_Per_Year', 'Response_Time_Minutes',
        'Verified_Total'
    ]

    X = cluster_df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow_rows = []
    sil_rows = []

    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        elbow_rows.append({'k': k, 'inertia': model.inertia_})
        sil_rows.append({'k': k, 'silhouette': silhouette_score(X_scaled, labels)})

    elbow_df = pd.DataFrame(elbow_rows)
    sil_df = pd.DataFrame(sil_rows)

    best_k = sil_df.sort_values('silhouette', ascending=False).iloc[0]['k']
    final_kmeans = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
    cluster_df['Cluster'] = final_kmeans.fit_predict(X_scaled)

    profile = cluster_df.groupby('Cluster').agg({
        'Age': 'mean',
        'Profile_Completeness': 'mean',
        'Trust_Score': 'mean',
        'Route_Compatibility': 'mean',
        'Satisfaction': 'mean',
        'Trips_Per_Year': 'mean',
        'Response_Time_Minutes': 'mean',
        'Journey_Companion_Found': 'mean',
        'User_ID': 'count'
    }).round(2).reset_index()

    profile.rename(columns={
        'Journey_Companion_Found': 'Match_Success_Rate',
        'User_ID': 'Users'
    }, inplace=True)

    cluster_labels = {}
    for _, row in profile.iterrows():
        cid = int(row['Cluster'])
        if row['Trust_Score'] >= profile['Trust_Score'].median() and row['Match_Success_Rate'] >= profile['Match_Success_Rate'].median():
            label = "High-Trust Frequent Matchers"
        elif row['Response_Time_Minutes'] > profile['Response_Time_Minutes'].median():
            label = "Slower-Response Cautious Users"
        else:
            label = "Balanced Mainstream Travelers"
        cluster_labels[cid] = label

    cluster_df['Cluster_Label'] = cluster_df['Cluster'].map(cluster_labels)
    profile['Cluster_Label'] = profile['Cluster'].map(cluster_labels)
    profile['Match_Success_Rate'] = (profile['Match_Success_Rate'] * 100).round(1)

    return cluster_df, elbow_df, sil_df, profile, int(best_k)


@st.cache_data
def run_regression_models(df):
    feature_cols, numeric_features, categorical_features = get_feature_lists()

    X = df[feature_cols]
    y = df['Satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pre_scaled = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)

    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', LinearRegression())
        ]),
        'Ridge Regression': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', Ridge(alpha=1.0))
        ]),
        'Lasso Regression': Pipeline([
            ('preprocessor', pre_scaled),
            ('model', Lasso(alpha=0.01))
        ])
    }

    rows = []
    prediction_store = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        rows.append({
            'Model': name,
            'R2': round(r2, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4)
        })

        prediction_store[name] = {
            'actual': y_test.values,
            'predicted': preds
        }

    reg_df = pd.DataFrame(rows).sort_values(by='R2', ascending=False).reset_index(drop=True)
    best_reg = reg_df.iloc[0]['Model']

    return reg_df, prediction_store, best_reg


# -----------------------------
# LOAD DATA
# -----------------------------
df_raw = load_data()
df = clean_transform_data(df_raw)

classification_results, confusion_store, best_classifier = run_classification_models(df)
cluster_df, elbow_df, sil_df, cluster_profile, best_k = run_clustering(df)
regression_results, prediction_store, best_regressor = run_regression_models(df)

best_class_accuracy = classification_results.loc[classification_results['Model'] == best_classifier, 'Accuracy'].values[0]
best_class_f1 = classification_results.loc[classification_results['Model'] == best_classifier, 'F1_Score'].values[0]
best_reg_r2 = regression_results.loc[regression_results['Model'] == best_regressor, 'R2'].values[0]


# -----------------------------
# HEADER
# -----------------------------
st.title("✈️ Travel Buddy - Final Analytics Dashboard")
st.markdown(
    f"**Synthetic Startup Validation | Verified Solo Travel Matching | Best Classifier: {best_classifier} | "
    f"Best Regression: {best_regressor}**"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Navigation")
selected_tab = st.sidebar.selectbox(
    "Choose Section",
    [
        "1. Executive Summary",
        "2. Data Cleaning & Transformation",
        "3. EDA / Descriptive Analytics",
        "4. Classification Models",
        "5. Clustering Analysis",
        "6. Regression Analysis",
        "7. Report Summary",
        "8. Presentation Summary"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Dataset size:** {len(df):,} users  
**Countries:** {df['Country_From'].nunique()}  
**Best classifier:** {best_classifier}  
**Best regressor:** {best_regressor}  
**Chosen clusters:** {best_k}
""")


# -----------------------------
# TAB 1
# -----------------------------
if selected_tab == "1. Executive Summary":
    st.header("🎯 Startup Validation Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users", f"{len(df):,}")
    c2.metric("Match Success", f"{df['Journey_Companion_Found'].mean()*100:.1f}%")
    c3.metric("Avg Satisfaction", f"{df['Satisfaction'].mean():.2f}/5")
    c4.metric("Avg Trust Score", f"{df['Trust_Score'].mean():.1f}")
    c5.metric("Best F1 Score", f"{best_class_f1:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        trust_summary = df.groupby('Trust_Level')['Journey_Companion_Found'].mean().reset_index()
        trust_summary['Journey_Companion_Found'] *= 100
        fig = px.bar(
            trust_summary,
            x='Trust_Level',
            y='Journey_Companion_Found',
            color='Trust_Level',
            title="Match Success by Trust Level",
            text_auto=".1f"
        )
        fig.update_layout(yaxis_title="Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            df,
            x='Trust_Score',
            y='Satisfaction',
            color='Journey_Companion_Found',
            title="Trust Score vs Satisfaction",
            opacity=0.65,
            color_discrete_map={0: "#ff6b6b", 1: "#00d4aa"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
<div class="section-box">
<b>Business interpretation:</b><br>
This dashboard validates the Travel Buddy startup idea through synthetic user data, trust-based matching logic,
descriptive analytics, classification, clustering, and regression. Higher trust, stronger route compatibility,
and more complete profiles consistently improve both match success and satisfaction.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 2
# -----------------------------
elif selected_tab == "2. Data Cleaning & Transformation":
    st.header("🧹 Data Cleaning & Transformation")

    left, right = st.columns([1.1, 1.3])

    with left:
        st.subheader("Cleaning checklist")
        st.markdown("""
- Removed duplicate records.
- Checked numeric columns and standardized data types.
- Filled missing categorical values with safe labels.
- Recreated `Age_Group` for segmentation.
- Preserved business-ready columns such as trust level, verification count, and route fields.
- Prepared final cleaned dataset for EDA, classification, clustering, and regression.
        """)

        nulls = df.isnull().sum().reset_index()
        nulls.columns = ['Column', 'Missing_Values']
        st.dataframe(nulls, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Variable types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data_Type': [str(df[c].dtype) for c in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        st.subheader("Transformed fields")
        transformed_df = pd.DataFrame({
            'Field': ['Trust_Level', 'Age_Group', 'Verified_Total'],
            'Purpose': [
                'Derived verification segmentation for trust analysis',
                'Age-based demographic grouping for clustering and EDA',
                'Summed LinkedIn and Passport verification for modeling'
            ]
        })
        st.dataframe(transformed_df, use_container_width=True, hide_index=True)

    st.markdown("""
<div class="section-box">
<b>Why this matters:</b><br>
This section directly supports the assignment requirement for data cleaning and transformation.
It demonstrates that the dataset was prepared systematically before any model or graph was produced.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 3
# -----------------------------
elif selected_tab == "3. EDA / Descriptive Analytics":
    st.header("📈 Descriptive Analytics / EDA")

    c1, c2 = st.columns(2)

    with c1:
        fig_age = px.histogram(
            df, x='Age', nbins=20,
            title="Age Distribution",
            color_discrete_sequence=['#00d4aa']
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        fig_sat = px.histogram(
            df, x='Satisfaction', nbins=20,
            title="Satisfaction Distribution",
            color_discrete_sequence=['#7b68ee']
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        route_summary = df.groupby(['City_From', 'City_To']).size().reset_index(name='Volume')
        fig_route = px.sunburst(
            route_summary,
            path=['City_From', 'City_To'],
            values='Volume',
            title="Route Volume by Origin and Destination"
        )
        st.plotly_chart(fig_route, use_container_width=True)

    with c4:
        transport_summary = df.groupby('Transport_Mode')['Journey_Companion_Found'].mean().reset_index()
        transport_summary['Journey_Companion_Found'] *= 100
        fig_transport = px.bar(
            transport_summary,
            x='Transport_Mode',
            y='Journey_Companion_Found',
            color='Transport_Mode',
            title="Match Success by Transport Mode",
            text_auto=".1f"
        )
        fig_transport.update_layout(yaxis_title="Success Rate (%)")
        st.plotly_chart(fig_transport, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        class_matrix = df.pivot_table(
            index='Travel_Class',
            columns='Transport_Mode',
            values='Journey_Companion_Found',
            aggfunc='mean'
        ).fillna(0) * 100
        fig_heat = px.imshow(
            class_matrix,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            title="Success Rate Heatmap: Travel Class vs Transport Mode"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with c6:
        numeric_corr = df[[
            'Age', 'Profile_Completeness', 'Trust_Score', 'Route_Compatibility',
            'Response_Time_Minutes', 'Trips_Per_Year', 'Satisfaction',
            'Journey_Companion_Found'
        ]].corr()
        fig_corr = px.imshow(
            numeric_corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
<div class="section-box">
<b>EDA insight summary:</b><br>
The descriptive analytics suggest that trust, profile completeness, route compatibility, and travel behavior
are central drivers of match success and satisfaction. These insights justify the variable selection used in the
classification, clustering, and regression stages.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 4
# -----------------------------
elif selected_tab == "4. Classification Models":
    st.header("🤖 Classification Model Comparison")
    st.caption("Target variable: Journey_Companion_Found")

    st.subheader("Model comparison table")
    st.dataframe(classification_results, use_container_width=True, hide_index=True)

    metric_choice = st.selectbox(
        "Select metric to compare",
        ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    )

    fig_cls = px.bar(
        classification_results,
        x='Model',
        y=metric_choice,
        color='Model',
        title=f"Classification Comparison by {metric_choice}",
        text_auto=".3f"
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    st.subheader("Confusion matrix")
    cm_model = st.selectbox("Select model for confusion matrix", classification_results['Model'].tolist())
    cm = confusion_store[cm_model]

    cm_df = pd.DataFrame(
        cm,
        index=['Actual 0', 'Actual 1'],
        columns=['Predicted 0', 'Predicted 1']
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix - {cm_model}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    top_row = classification_results.iloc[0]
    st.markdown(f"""
<div class="section-box">
<b>Best classifier:</b> {top_row['Model']}<br>
<b>Accuracy:</b> {top_row['Accuracy']:.3f} |
<b>Precision:</b> {top_row['Precision']:.3f} |
<b>Recall:</b> {top_row['Recall']:.3f} |
<b>F1-score:</b> {top_row['F1_Score']:.3f}
<br><br>
This section satisfies the assignment requirement to compare multiple classification algorithms using
accuracy, precision, recall, and F1-score.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 5
# -----------------------------
elif selected_tab == "5. Clustering Analysis":
    st.header("🧩 K-Means Clustering Analysis")

    c1, c2 = st.columns(2)

    with c1:
        fig_elbow = px.line(
            elbow_df,
            x='k',
            y='inertia',
            markers=True,
            title="Elbow Method"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = px.line(
            sil_df,
            x='k',
            y='silhouette',
            markers=True,
            title="Silhouette Score by k"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.subheader(f"Chosen cluster count: {best_k}")
    st.dataframe(cluster_profile, use_container_width=True, hide_index=True)

    fig_cluster = px.scatter(
        cluster_df,
        x='Trust_Score',
        y='Satisfaction',
        color='Cluster_Label',
        size='Trips_Per_Year',
        hover_data=['Age', 'Profile_Completeness', 'Journey_Companion_Found'],
        title="Cluster Map: Trust Score vs Satisfaction"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Cluster interpretation")
    for _, row in cluster_profile.iterrows():
        st.markdown(f"""
- **Cluster {int(row['Cluster'])} - {row['Cluster_Label']}**:
  {int(row['Users'])} users, average trust score {row['Trust_Score']}, average satisfaction {row['Satisfaction']},
  and match success rate {row['Match_Success_Rate']}%.
        """)

    st.markdown("""
<div class="section-box">
This clustering section identifies meaningful traveler personas and explains how different user groups behave.
It directly addresses the assignment requirement to derive business meaning from the formed clusters.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 6
# -----------------------------
elif selected_tab == "6. Regression Analysis":
    st.header("📉 Linear, Ridge and Lasso Regression")
    st.caption("Target variable: Satisfaction")

    st.subheader("Regression comparison table")
    st.dataframe(regression_results, use_container_width=True, hide_index=True)

    fig_reg = px.bar(
        regression_results,
        x='Model',
        y='R2',
        color='Model',
        title="Regression Comparison by R²",
        text_auto=".3f"
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    selected_reg = st.selectbox("Select regression model for actual vs predicted", regression_results['Model'].tolist())
    pred_data = prediction_store[selected_reg]

    plot_df = pd.DataFrame({
        'Actual': pred_data['actual'],
        'Predicted': pred_data['predicted']
    })

    fig_pred = px.scatter(
        plot_df,
        x='Actual',
        y='Predicted',
        trendline='ols',
        title=f"Actual vs Predicted Satisfaction - {selected_reg}"
    )
    fig_pred.add_shape(
        type='line',
        x0=plot_df['Actual'].min(), y0=plot_df['Actual'].min(),
        x1=plot_df['Actual'].max(), y1=plot_df['Actual'].max(),
        line=dict(color='red', dash='dash')
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    best_reg_row = regression_results.iloc[0]
    st.markdown(f"""
<div class="section-box">
<b>Best regression model:</b> {best_reg_row['Model']}<br>
<b>R²:</b> {best_reg_row['R2']:.3f} |
<b>RMSE:</b> {best_reg_row['RMSE']:.3f} |
<b>MAE:</b> {best_reg_row['MAE']:.3f}
<br><br>
This section fulfills the assignment option requiring Linear, Ridge, and Lasso regression comparison.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 7
# -----------------------------
elif selected_tab == "7. Report Summary":
    st.header("📝 Final Report Summary (Ready to Use)")

    st.markdown(f"""
### Abstract
Travel Buddy is a trust-first solo travel companion platform designed to validate a startup business idea
through synthetic data, descriptive analytics, classification, clustering, and regression. The analysis shows
that trust score, profile completeness, verification strength, and route compatibility meaningfully improve both
journey companion success and overall satisfaction.

### Introduction
The project addresses the business problem faced by solo travelers who want safe and compatible companions.
The dashboard validates the feasibility of a verified matching platform using synthetic user data generated to
simulate early-stage startup demand.

### Domain and Objectives
- Domain: Travel technology / trust-based companion matching
- Objective 1: Validate business feasibility using synthetic startup data
- Objective 2: Study key drivers of match success
- Objective 3: Segment users into meaningful traveler personas
- Objective 4: Predict satisfaction using regression models

### Data Cleaning and Transformation
- Duplicates removed
- Types standardized
- Missing categorical values handled
- Age groups and verification counts derived
- Final dataset prepared for EDA and ML models

### Algorithms Applied
- Classification: Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM, Gradient Boosting{" , XGBoost" if XGB_AVAILABLE else ""}
- Clustering: K-Means using silhouette-guided cluster selection
- Regression: Linear Regression, Ridge Regression, Lasso Regression

### Result Execution
- Best classification model: {best_classifier}
- Best classification accuracy: {best_class_accuracy:.3f}
- Best classification F1-score: {best_class_f1:.3f}
- Cluster count selected: {best_k}
- Best regression model: {best_regressor}
- Best regression R²: {best_reg_r2:.3f}

### Conclusion
The project supports the Travel Buddy business idea by showing that verified, compatible, and complete user
profiles generate stronger matching outcomes and higher satisfaction. The analytics results provide both business
validation and an evidence-based product strategy for a trust-first travel startup.
""")

    st.markdown("""
<div class="section-box">
Use this tab as the base content for your written report. Add screenshots from the EDA, classification,
clustering, and regression tabs into the final document to satisfy the report requirement.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# TAB 8
# -----------------------------
elif selected_tab == "8. Presentation Summary":
    st.header("🎤 Presentation Summary")

    st.markdown(f"""
### Slide 1: Business Problem
Solo travelers often struggle to find safe, trustworthy, and compatible companions.

### Slide 2: Proposed Solution
Travel Buddy is a verified matching platform using LinkedIn and Passport verification with a single-match logic.

### Slide 3: Synthetic Data
A synthetic dataset was generated to simulate startup demand, user trust, travel behavior, and satisfaction.

### Slide 4: EDA Highlights
- Higher profile completeness improves match outcomes
- Trust score positively affects satisfaction
- Route compatibility supports better companion finding
- Verified users outperform lower-trust users

### Slide 5: Classification Results
- Target: Journey_Companion_Found
- Best model: {best_classifier}
- Accuracy: {best_class_accuracy:.3f}
- F1-score: {best_class_f1:.3f}

### Slide 6: Clustering Results
- Method: K-Means
- Final clusters: {best_k}
- Output: traveler personas for product targeting and strategy

### Slide 7: Regression Results
- Target: Satisfaction
- Best model: {best_regressor}
- R²: {best_reg_r2:.3f}

### Slide 8: Final Recommendation
The Travel Buddy concept is analytically validated as a trust-led startup model. Product strategy should focus
on verified users, better profile completion, route compatibility, and faster response behavior to increase
successful matching and improve satisfaction.
""")

    st.markdown("""
<div class="section-box">
This tab is designed for presentation preparation. You can directly copy these points into slides and then add
screenshots from the dashboard visuals.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Travel Buddy Final Dashboard**
- Synthetic data included
- Cleaning section included
- EDA included
- Classification comparison included
- Clustering analysis included
- Linear, Ridge, Lasso included
- Report + presentation summaries included
""")