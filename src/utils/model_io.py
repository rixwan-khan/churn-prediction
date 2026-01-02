# src/utils/model_io.py
"""
Purpose: Centralized functions for saving and loading ML models.
"""
import joblib
from pathlib import Path
from src.utils.paths import MODELS_DIR
from src.utils.logger import get_logger


# logger configuration
logger =  get_logger(
    log_filename='model_io.log',
    log_subdir='utils'
)

def save_model(model, model_name:str):
    """
    Purpose : Save a trained ML model to disk in a structured directory.
    Args    :
             model: Trained model object (sklearn, XGBoost, etc.)
             model_name (str): Name to save the model under

    Returns :
             model_path (Path): Full path where the model is saved

    Notes   :
        - Creates the directory if it doesn't exist.
        - Saves model in .pkl format using joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f'{model_name}.pkl'
    joblib.dump(model, model_path)

    return model_path

def load_model(model_name:str):
    """
    Purpose : Load a previously saved ML model.
    Args    : model_name (str): Name of the model file to load.
    Returns : model: Loaded model object ready for inference or evaluation
    Notes   : model is saved in MODELS_DIR in .pkl format
    """
    model_path = MODELS_DIR / f'{model_name}.pkl'

    if not model_path.exists():
        logger.error(f'Model file not found: {model_path}')
        raise FileNotFoundError(f'No model found at {model_path}')
    
    logger.info(f"Loaded model '{model_name}' from: {model_path}")

    return joblib.load(model_path)


def load_best_model(model_name:str= None, best_models:dict=None):
    """
    Load the best ML model. 

    Args:
        - model_name (str, optional): Name of a specific model to load. If None, all best models are loaded.
        - best_models (dict, optional): Dictionary of model names to load. Example:
            {'logistic_regression': 'logistic_regression',
             'xgboost': 'xgboost'}

    Returns:
        model object or dict: Loaded model(s). If `model_name` is provided, returns single model. 
                              If None, returns a dict of all best models.
    """
    # Default best model
    if best_models is None:
        best_models = {
            'logistic_regression': 'logistic_regression',
            'xgboost': 'xgboost'
        }
    
    # Load single model
    if model_name:
        if model_name not in best_models:
            logger.error(f"Model '{model_name}' not found in best_models." )
            raise ValueError(f"Model '{model_name}' not in best_models")
        return load_model(best_models[model_name])
    
    # Load all models
    loaded_models = {}
    for name, path in best_models.items():
        loaded_models[name] = load_model(path)

    logger.info(f"Loaded all best models: {list(loaded_models.keys())}")
    return load_model(model_name)