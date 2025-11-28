"""
RUN PREPROCESSING PIPELINE
--------------------------
This script runs the full preprocessing pipeline:
- Loads raw data
- Cleans and preprocesses data
- Logs all steps and prints summaries
- Saves cleaned dataset
"""

# -------------------------------
# Import required modules and settings
# -------------------------------
from src.config import PROCESSED_DATA_PATH, CLEAN_DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS
from src.preprocessing.preprocessing import preprocess_pipeline
from src.utils.logger import get_logger

# -------------------------------
# Initialize logger
# -------------------------------
# Logger will write messages to a file 'preprocessing.log'
# This allows us to keep a permanent record of what happened in the pipeline
logger = get_logger("preprocessing.log")

# -------------------------------
# Main function
# -------------------------------
def main():
    try:
        # -------------------------------
        # Step 1: Start message
        # -------------------------------
        # Inform user and log that preprocessing has started
        print("=== Preprocessing pipeline started ===\n")
        logger.info("=== Preprocessing pipeline started ===")

        # -------------------------------
        # Step 2: Log input and output paths
        # -------------------------------
        # Print and log which files we are reading and saving
        print(f"Input file: {PROCESSED_DATA_PATH}")
        print(f"Output file: {CLEAN_DATA_PATH}\n")
        logger.info(f"Input file: {PROCESSED_DATA_PATH}")
        logger.info(f"Output file: {CLEAN_DATA_PATH}")

        # -------------------------------
        # Step 3: Run the preprocessing pipeline
        # -------------------------------
        # This will:
        # 1. Load the data
        # 2. Remove duplicates
        # 3. Convert datatypes
        # 4. Handle missing values
        # 5. Detect & flag outliers
        # 6. Scale numeric features (optional)
        # 7. Save cleaned dataset
        # Pass the logger so every step logs messages to the file
        df_cleaned = preprocess_pipeline(
            input_path=PROCESSED_DATA_PATH,
            output_path=CLEAN_DATA_PATH,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            scale=True,   # True = standardize numeric columns
            logger=logger # Pass logger for logging inside preprocessing.py
        )

        # -------------------------------
        # Step 4: Completion message
        # -------------------------------
        # Inform user and log that pipeline finished successfully
        print("\nPreprocessing pipeline completed successfully")
        print(f"Final cleaned dataset shape: {df_cleaned.shape}")
        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Final cleaned dataset shape: {df_cleaned.shape}")

    except Exception as e:
        # -------------------------------
        # Step 5: Error handling
        # -------------------------------
        # Print and log error if pipeline fails
        print(f"Preprocessing pipeline failed: {e}")
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise e  # Raise the exception so you can see full traceback in console

# -------------------------------
# Script entry point
# -------------------------------
# When this file is run directly, Python will call main()
if __name__ == "__main__":
    main()
