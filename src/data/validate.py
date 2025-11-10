# src/data/validate.py
import pandas as pd
import pandera.pandas as pa
from schema import ChurnSchema, RAW_DATA_PATH

def main():
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    try:
        validated_df = ChurnSchema.validate(df, lazy=True)
        print("All data passed validation!")
    except pa.errors.SchemaErrors as e:
        print("VALIDATION ERRORS FOUND:")
        print(e.failure_cases)  # sirf real failed rows show kare

if __name__ == "__main__":
    main()