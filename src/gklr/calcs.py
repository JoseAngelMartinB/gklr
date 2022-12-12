"""GKLR calcs module."""
from abc import ABC, abstractmethod

import numpy as np

from .kernel_matrix import KernelMatrix

class Calcs(ABC):
    """Base Calcs class object."""

    def __init__(self, K: KernelMatrix) -> None:
        """Constructor.

        Args:
            K: KernelMatrix object.
        """
        self.K = K

    @abstractmethod
    def calc_probabilities(self, alpha):
        """Calculate the probabilities for each alternative."""
        return

    @abstractmethod
    def log_likelihood(self, alpha, P, choice_indices):
        """Calculate the log-likelihood of the model for the given parameters.
        """
        return

    @abstractmethod
    def log_likelihood_and_gradient(self, alpha):
        """Calculate the log-likelihood of the model and its gradient for
        the given parameters.
        """
        return

    @abstractmethod
    def calc_f(self, alpha):
        """Calculate the value of utility function for each alternative for each row
        of the dataset."""
        return

    def calc_Y(self, f: np.ndarray) -> np.ndarray:
        """Calculate the auxiliary matrix `Y` that contains the exponentiated
        values of the matrix `f`.

        Args:
            f: The matrix of utility function values for each alternative for each
                row of the dataset.

        Returns:
            The auxiliary matrix `Y` that contains the exponentiated values of the
                matrix `f`.
        """
        return np.exp(f)

    @abstractmethod
    def calc_G(self, Y):
        """Calculate the auxiliary matrix `G` and its derivative."""
        return

    def calc_P(self,
               Y: np.ndarray,
               G: np.ndarray,
               G_j: np.ndarray
    ) -> np.ndarray:
        """Calculate the matrix of probabilities for each alternative for each row
        of the dataset.

        Args:
            Y: The auxiliary matrix `Y` that contains the exponentiated values of
                the matrix `f`.
            G: The auxiliary matrix `G`.
            G_j: The derivative of the auxiliary matrix `G`.

        Returns:
            The matrix of probabilities for each alternative for each row of the
                dataset. Each column corresponds to an alternative and each row
                to a row of the dataset. The sum of the probabilities for each
                row is 1.


        """
        return (Y*G_j)/G
