# src/data/validate.py
import os
import warnings
import pandas as pd
from pandera.errors import SchemaErrors
from src.data.schema import ChurnSchema, RAW_DATA_PATH
from src.utils.io import load_csv, save_csv

warnings.simplefilter(action="ignore", category=FutureWarning)


def validate_data(df, schema, validated_path, failed_path):
    """Validate dataframe against schema. Save valid and invalid rows separately."""
    os.makedirs(os.path.dirname(validated_path), exist_ok=True)
    os.makedirs(os.path.dirname(failed_path), exist_ok=True)

    try:
        # Validate with lazy=True to collect all errors
        validated_df = schema.validate(df, lazy=True)
        print("✅ All rows passed validation.")
        save_csv(validated_df, validated_path)
        failed_rows = pd.DataFrame()  # Empty DataFrame

    except SchemaErrors as err:
        print("⚠️ Validation errors found!")

        # Get the failing rows based on the error indices
        error_df = err.failure_cases
        failed_rows = df.iloc[error_df["index"].unique()]
        valid_rows = df.drop(error_df["index"].unique())

        # Save both valid and failed rows
        save_csv(valid_rows, validated_path)
        save_csv(failed_rows, failed_path)

        print(f"✅ Valid rows saved to: {validated_path}")
        print(f"❌ Failed rows saved to: {failed_path}")

        print("\nSummary of validation issues:")
        print(error_df.head(10))

        return valid_rows, failed_rows

    return validated_df, failed_rows


if __name__ == "__main__":
    print(f"\n📂 Loading data from: {RAW_DATA_PATH}")
    df = load_csv(RAW_DATA_PATH)

    validated_path = os.path.join("data", "processed", "validated_df.csv")
    failed_path = os.path.join("data", "logs", "failed_rows.csv")

    validate_data(df, ChurnSchema, validated_path, failed_path)
