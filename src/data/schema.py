# src/data/schema.py
import os
import pandera as pa
from pandera import Column, Check
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Import centralized path from paths.py
from ..utils.paths import RAW_DATA_PATH

ChurnSchema = pa.DataFrameSchema(
    columns={
        "gender": Column(str, Check(lambda s: s.isin(["Male", "Female"])), nullable=False),
        "SeniorCitizen": Column(int, Check(lambda s: s.isin([0, 1])), nullable=False),
        "Partner": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        "Dependents": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        "tenure": Column(int, Check(lambda s: s >= 0), nullable=False),
        "PhoneService": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        "InternetService": Column(str, Check(lambda s: s.isin(["DSL", "Fiber optic", "No"])), nullable=False),
        "Contract": Column(str, Check(lambda s: s.isin(["Month-to-month", "One year", "Two year"])), nullable=False),
        "PaperlessBilling": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        "MonthlyCharges": Column(float, Check(lambda s: s >= 0), nullable=False),
        "Churn": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
    },
    coerce=True,  # Automatically convert datatypes if needed
    strict=False  # Allow extra columns if present
)
