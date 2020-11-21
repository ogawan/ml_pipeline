import sys
sys.path.insert(0, "/work/ml_pipeline/regression_models/")
import regression_models
import numpy as np
from sklearn.model_selection import train_test_split
import pipeline
from processing.data_management import load_dataset, save_pipeline
from config import config
from regression_models import __version__ as _version

import logging

_logger = logging.getLogger(__name__)

def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    print("DATA:")
    print(data)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  # we are setting the seed here
    print("Complete Splitting data")

    pipeline.chem_pipe.fit(X_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.chem_pipe)
    
if __name__ == "__main__":
    run_training()
