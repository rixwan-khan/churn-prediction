# src/validation/schema.py
import pandera as pa
from pandera import Column, Check  # Import Column and Check for defining schema rules
import warnings   # Import warnings to handle deprecated/future warnings

# Ignore future warnings from libraries
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define schema for Telco Churn dataset
ChurnSchema = pa.DataFrameSchema(
    columns={
        
        # Gender must be 'Male' or 'Female'
        "gender": Column(str, Check(lambda s: s.isin(["Male", "Female"])), nullable=False),
        
         # SeniorCitizen must be 0 or 1
        "SeniorCitizen": Column(int, Check(lambda s: s.isin([0, 1])), nullable=False),
        
        # Partner column must be 'Yes' or 'No'
        "Partner": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        
        # Dependents must be 'Yes' or 'No'
        "Dependents": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        
        # Tenure must be a non-negative integer
        "tenure": Column(int, Check(lambda s: s >= 0), nullable=False),
        
        # PhoneService must be 'Yes' or 'No'
        "PhoneService": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        
        # InternetService must be one of specified options
        "InternetService": Column(str, Check(lambda s: s.isin(["DSL", "Fiber optic", "No"])), nullable=False),
        
        # Contract type must match allowed values
        "Contract": Column(str, Check(lambda s: s.isin(["Month-to-month", "One year", "Two year"])), nullable=False),
        
        # PaperlessBilling must be 'Yes' or 'No'
        "PaperlessBilling": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
        
        # MonthlyCharges must be non-negative float
        "MonthlyCharges": Column(float, Check(lambda s: s >= 0), nullable=False),
        
        # Churn must be 'Yes' or 'No'
        "Churn": Column(str, Check(lambda s: s.isin(["Yes", "No"])), nullable=False),
    },
    coerce=True,  # Automatically convert datatypes if needed
    strict=False  # Allow extra columns if present
)
