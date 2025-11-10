# src/data/schema.py
import os
import pandas as pd
import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check

# To ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# File path
RAW_DATA_PATH = os.path.join("data", "raw", "telco-customer-churn-raw.csv")

# Defining Schema
ChurnSchema = DataFrameSchema(
    {
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "SeniorCitizen": Column(int, Check.isin([0, 1]), nullable=False),
        "Partner": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "Dependents": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "tenure": Column(int, Check.ge(0), nullable=False),
        "PhoneService": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"]), nullable=False),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"]), nullable=False),
        "MonthlyCharges": Column(float, Check.ge(0), nullable=False),
        "Churn": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    }
)