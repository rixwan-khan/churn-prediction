# Runs the full preprocessing pipeline and logs all steps.
from src.config import PROCESSED_DATA_PATH, CLEAN_DATA_PATH, NUMERIC_COLS, CATEGORICAL_COLS
from src.preprocessing.preprocessing import preprocess_pipeline
from src.utils.logger import get_logger

# -------------------------------
# Initialize logger
# -------------------------------
logger = get_logger("preprocessing.log")

# -------------------------------
# Run pipeline
# -------------------------------
def main():
    try:
        logger.info("=== Preprocessing pipeline started ===")

        logger.info(f"Input file: {PROCESSED_DATA_PATH}")
        logger.info(f"Output file: {CLEAN_DATA_PATH}")

        # Run preprocessing
        df_cleaned = preprocess_pipeline(
            input_path=PROCESSED_DATA_PATH,
            output_path=CLEAN_DATA_PATH,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            scale=True  # optional feature scaling
        )

        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Final cleaned dataset shape: {df_cleaned.shape}")

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise e


# -------------------------------
# Script entry point
# -------------------------------
if __name__ == "__main__":
    main()
