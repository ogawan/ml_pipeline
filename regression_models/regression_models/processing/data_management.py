import sys
sys.path.insert(0, "/work/ml_pipeline/regression_models/")

import pandas as pd
import sklearn.externals.joblib as joblib
from sklearn.pipeline import Pipeline
from config import config
from regression_models import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return _data

def save_pipeline(*, pipeline_to_persist, model_name) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """
    
    model_name = model_name + "_regression"
    
    # Prepare versioned save file name
    save_file_name = f"{model_name}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_file_name

    #remove_old_pipelines(files_to_keep=save_file_name)
    joblib.dump(pipeline_to_persist, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = config.TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
