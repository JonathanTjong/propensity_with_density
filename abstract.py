from abc import ABC, abstractmethod

import numpy as np


class AbstractOverlapEstimator(ABC):
    """
    Abstract class for overlap estimators.
    Defines the interface for overlap estimators.

    Example usage: class MyOverlapEstimator(AbstractOverlapEstimator) ...
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the overlap estimator to the data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether each data point is in the overlap region.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.

        Returns:
            np.ndarray: Array of shape (n_samples, ) with 0 or 1 for each data point.
                        0 means the data point is not in the overlap region,
                        and 1 means the data point is in the overlap region.
        """
        pass

    #@abstractmethod
    def get_overlap_region(self) -> object:
        """Return the overlap region identified by the overlap estimator.
        #TODO Have to figure out what representation to use for the overlap region.

        Returns:
            object: The overlap region.
        """
        pass

    @abstractmethod
    def score(
        self, X: np.ndarray, y: np.ndarray, score_type: str = "accuracy"
    ) -> float:
        """Compute the score of the model with X and y as test data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the test data points.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
            score_type (str, optional): The type of score to compute. Defaults to "accuracy". # TODO Think about other scores.

        Returns:
            float: The score of the model.
        """
        pass

    #@abstractmethod
    def get_params(self) -> dict:
        """Return the parameters of the overlap estimator. #TODO maybe not needed

        Returns:
            dict: The parameters of the overlap estimator.
        """
        pass

    #@abstractmethod
    def set_params(self, **params) -> None:
        """Set the parameters of the overlap estimator. #TODO maybe not needed

        Args:
            **params: The parameters to set.
        """
        pass
