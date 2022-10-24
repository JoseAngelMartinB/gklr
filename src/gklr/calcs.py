"""GKLR calcs module."""

import numpy as np

from .kernel_matrix import KernelMatrix

class Calcs():
    """Base Calcs class object."""

    def __init__(self, K: KernelMatrix) -> None:
        """Constructor.

        Args:
            K: KernelMatrix object.
        """
        self.K = K

    def calc_probabilities(self):
        """Calculate the probabilities for each alternative."""
        pass

    def log_likelihood(self):
        """Calculate the log-likelihood of the KLR model for the given parameters."""
        pass

    def calc_f(self):
        """Calculate the value of utility function for each alternative for each row
        of the dataset."""
        pass

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

    def calc_G(self):
        """Calculate the auxiliary matrix `G` and its derivative."""
        pass

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
                dataset.
        """
        return (Y*G_j)/G