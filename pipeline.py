from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from regression_model.processing import preprocessors as pp

import logging


_logger = logging.getLogger(__name__)

price_pipe = Pipeline(
    [
        (
            "SmilestoDescriptors",
            pp.SmilestoDescriptors(mode="rdkit"),
        ),
         (
            "DropChollinearityVif",
            pp.DropChollinearityVif(threshold=10),
        ),       
        ("scaler", MinMaxScaler()),
        ("Linear_model", Lasso(alpha=0.005, random_state=0)),
    ]
)
