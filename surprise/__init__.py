from . import dump, model_selection

# Import similarities module (Cython extension)
try:
    from . import similarities  # type: ignore
except ImportError:
    # Handle case where Cython extension is not built
    similarities = None
from .builtin_datasets import get_dataset_dir
from .dataset import Dataset
from .prediction_algorithms import (
    NMF,
    SVD,
    AlgoBase,
    BaselineOnly,
    CoClustering,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NormalPredictor,
    Prediction,
    PredictionImpossible,
    SlopeOne,
    SVDpp,
)
from .reader import Reader
from .trainset import Trainset

__all__ = [
    "AlgoBase",
    "NormalPredictor",
    "BaselineOnly",
    "KNNBasic",
    "KNNWithMeans",
    "KNNBaseline",
    "SVD",
    "SVDpp",
    "NMF",
    "SlopeOne",
    "CoClustering",
    "PredictionImpossible",
    "Prediction",
    "Dataset",
    "Reader",
    "Trainset",
    "dump",
    "KNNWithZScore",
    "get_dataset_dir",
    "model_selection",
    "similarities",
]

__version__ = "1.1.4"
