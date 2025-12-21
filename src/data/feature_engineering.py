# src/data/feature_engineering.py

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

#========================================================
# ADD PROJECT ROOT TO PATH
#========================================================

#Allows Python to recognize the src/ folder modules when running scripts
sys.path.append(os.path.abspath(".."))

from src.utils.paths import PROCESSED_DATA_DIR, DATA_DIR
from src.utils.io import load_csv, save_csv
from src.utils.logger import get_logger

#========================================================
# INITIALIZE LOGGER
#========================================================
logger = get_logger("feature_engineering.log", log_subdir="feature_engineering")


#========================================================
# FEATURE ENGINEERING FUNCTION
#========================================================
def engineer_features(df):
    """
    Perform custom feature engineering and encoding:
    - Add new features
    - Encode categorical columns
    """
    df = df.copy()

    #=================
    # ENSURE NUMERIC TYPES
    #=================
    numeric_cols = ['TotalCharges', 'MonthlyCharges', 'tenure']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['tenure'] = df['tenure'].fillna(1)  # avoid division by zero
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())

    #=================
    # CUSTOM FEATURES
    #=================
    # Count of all services subscribed by customer
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]

    # count 'Yes' values per row
    df['service_count'] = df[service_cols].eq('Yes').sum(axis=1)

    # Bucket MonthlyCharges into categories for non-linear effects
    df['charges_bucket'] = pd.cut(
        df['MonthlyCharges'],
        bins=[0, 35, 70, 100, 150],
        labels=['Low', 'Medium', 'High', 'Very High']
    ).cat.codes  # convert categorical to numeric

    # Average charges per month, handling tenure=0
    df['charges_per_month'] = np.where(
        df['tenure'] != 0,
        df['TotalCharges'] / df['tenure'],
        0
    )

    # Active user flag: has internet and streaming TV subscription
    df['is_active_user'] = (
        (df['InternetService'] != 'No') & (df['StreamingTV'] == 'Yes')
    ).astype(int)

    # Senior citizen flag: convert boolean/integer directly
    df['is_senior'] = df['SeniorCitizen'].astype(int)
    
     # Non-linear tenure effect
    df['tenure_sqr'] = df['tenure'] ** 2
    
     # Flag high monthly charges above median
    df['high_bill_flag'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

     # Drop customerID (identifier, not predictive)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

     # Map target variable Churn to numeric
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    #========================================================
    # ENCODING
    #========================================================
    ordinal_cols = ['Contract']
    ordinal_mapping = {'Contract': ['Month-to-month', 'One year', 'Two year']}
    for col, order in ordinal_mapping.items():
        encoder = OrdinalEncoder(categories=[order])
        df[col + "_encoded"] = encoder.fit_transform(df[[col]])
    df = df.drop(columns=ordinal_cols)

    nominal_cols = [c for c in df.select_dtypes(include="object").columns if c not in ['Churn']]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    logger.info("Feature engineering completed with %d columns", df.shape[1])
    return df


#========================================================
# BASELINE FEATURE SELECTION FUNCTION
#========================================================
def baseline_feature_selection(df):
    """
    Perform baseline feature reduction:
    - Remove zero-variance columns
    - Remove highly correlated columns
    """
    df = df.copy()
    target_col = 'Churn'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    #========================================================
    # ZERO-VARIANCE FILTERING
    #========================================================
    zero_var_cols = [c for c in X.columns if X[c].nunique() == 1]
    X = X.drop(columns=zero_var_cols)
    logger.info("Zero-variance features removed: %d columns left", X.shape[1])

    #========================================================
    # HIGHLY CORRELATED FEATURE FILTERING
    #========================================================
    X_numeric = X.select_dtypes(include=[np.number])  # only numeric columns for correlation
    corr_matrix = X_numeric.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    X = X.drop(columns=to_drop)
    logger.info("Highly correlated features removed: %d columns left", X.shape[1])

    df_final = X.join(y)
    return df_final


#========================================================
# MAIN FUNCTION
#========================================================
def main():
    #=================
    # LOAD PROCESSED DATA
    #=================
    df = load_csv(PROCESSED_DATA_DIR / "cleaned_telco_churn.csv")
    logger.info("Loaded cleaned dataset with %d rows and %d columns", df.shape[0], df.shape[1])

    #=================
    # FEATURE ENGINEERING
    #=================
    df_fe = engineer_features(df)

    #=================
    # BASELINE FEATURE SELECTION
    #=================
    df_final = baseline_feature_selection(df_fe)

    #=================
    # SAVE FEATURED DATASET
    #=================
    FEATURED_DIR = DATA_DIR / "04_featured"
    os.makedirs(FEATURED_DIR, exist_ok=True)
    save_csv(df_final, FEATURED_DIR / "featured_telco_churn.csv")
    logger.info("Saved featured dataset at %s", FEATURED_DIR / "featured_telco_churn.csv")

    print("Feature Engineering + Baseline Selection Completed")
    print("Final dataset shape:", df_final.shape)


#=================
# ENTRY POINT
#=================
if __name__ == "__main__":
    main()
