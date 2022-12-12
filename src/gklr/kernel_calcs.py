"""GKLR kernel_calcs module."""
from typing import Optional, Tuple

import numpy as np

from .logger import *
from .kernel_matrix import KernelMatrix
from .kernel_utils import *
from .calcs import Calcs

class KernelCalcs(Calcs):
    """Main calculations for the Kernel Logistic Regression (KLR) model."""
    
    def __init__(self, K: KernelMatrix) -> None:
        """Constructor.

        Args:
            K: KernelMatrix object.
        """
        super().__init__(K)

    def calc_probabilities(self, alpha: np.ndarray) -> np.ndarray:
        """Calculate the probabilities for each alternative.

        Obtain the probabilities for each alternative for each row of the
        dataset.

        Args:
            alpha: The vector of parameters.

        Returns:
            A matrix of probabilities for each alternative for each row of the
                dataset. Each column corresponds to an alternative and each row
                to a row of the dataset. The sum of the probabilities for each
                row is 1.
        """
        f = self.calc_f(alpha)
        Y = self.calc_Y(f)
        G, G_j = self.calc_G(Y)
        P = self.calc_P(Y, G, G_j)
        return P

    def log_likelihood(self,
                       alpha: np.ndarray,
                       P: Optional[np.ndarray] = None,
                       choice_indices: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the log-likelihood of the KLR model for the given parameters.

        Args:
            alpha: The vector of parameters.
            P: The matrix of probabilities of each alternative for each row of 
                the dataset. If None, the probabilities are calculated.
                Default: None.

        Returns:
            The log-likelihood of the KLR model for the given parameters.
        """
        if P is None:
            P = self.calc_probabilities(alpha)
        else:
            if P.shape != (self.K.get_num_rows(), self.K.get_num_alternatives()):
                m = (f"P has {P.shape} dimensions, but the dataset has"
                    f" {self.K.get_num_rows()} rows and {self.K.get_num_alternatives()}"
                    " alternatives.")
                logger_error(m)
                raise ValueError(m)
        if choice_indices is None:
            choice_indices = self.K.get_choices_indices()
        else:
            if len(choice_indices) != P.shape[0]:
                m = (f"choice_indices has {len(choice_indices)} elements, but P"
                    " has {P.shape[0]} rows.")
                logger_error(m)
                raise ValueError(m)
        
        log_P = np.log(P)
        log_likelihood = np.sum(log_P[np.arange(len(log_P)), choice_indices]) 
        return log_likelihood

    def log_likelihood_and_gradient(self,
                                    alpha: np.ndarray,
                                    pmle: Optional[str] = None,
                                    pmle_lambda: float = 0
    ) -> Tuple[float, np.ndarray]:
        """Calculate the log-likelihood of the KLR model and its gradient for
        the given parameters.

        Args:
            alpha: The vector of parameters.
            pmle: It specifies the type of penalization for performing a penalized
                maximum likelihood estimation.  Default: None.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
                 Default: 0.
        
        Returns:
            A tuple with the log-likelihood of the KLR model and its gradient for
                the given parameters.
        """
        P = self.calc_probabilities(alpha)

        # Log-likelihood
        log_likelihood = self.log_likelihood(alpha, P)

        # Gradient
        Z = self.K.get_choices_matrix()
        grad_penalization = 0
        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            grad_penalization = self.tikhonov_penalty_gradient(alpha, pmle_lambda)
        else:
            msg = f"ERROR. {pmle} is not a valid value for the penalization method `pmle`."
            logger_error(msg)
            raise ValueError(msg)

        H = grad_penalization + P - Z

        gradient = np.ndarray((self.K.get_num_rows(), 0), dtype=DEFAULT_DTYPE)
        for alt in range(0,self.K.get_num_alternatives()):
            gradient_alt = self.K.dot(H[:, alt], index=alt)
            gradient_alt = (gradient_alt / H.shape[0]).reshape((self.K.get_num_rows(),1))
            gradient = np.concatenate((gradient, gradient_alt), axis=1)

        gradient = gradient.reshape(self.K.get_num_rows() * self.K.get_num_alternatives())
        return (log_likelihood, gradient)

    def calc_f(self, alpha: np.ndarray) -> np.ndarray:
        """Calculate the value of utility function for each alternative for each row
        of the dataset.

        Args:
            alpha: The vector of parameters.

        Returns:
            A matrix where each row corresponds to the utility of each alternative
                for each row of the dataset.
        """
        f = np.ndarray((self.K.get_num_rows(), 0), dtype=DEFAULT_DTYPE)
        for alt in range(0,self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt].copy().reshape(self.K.get_num_cols(), 1)  # Get only the column for alt
            f_alt = self.K.dot(alpha_alt, index=alt)
            f = np.concatenate((f, f_alt), axis=1)
        return f

    def calc_G(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the auxiliary matrix `G` and its derivative.

        Args:
            Y: # TODO

        Returns:
            A tuple with the auxiliary matrix `G` and its derivative.
        """
        # Implementation for KLR
        G = np.sum(Y, axis=1).reshape((Y.shape[0], 1))
        # Compute G_j, the derivative of G with respecto to the variable Y_j
        G_j = np.ones_like(Y)
        return (G, G_j)

    def tikhonov_penalty(self,
                         alpha: np.ndarray,
                         pmle_lambda: float
    ) -> float:
        """Calculate the Tikhonov penalty for the given parameters.

        Args:
            alpha: The vector of parameters.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.

        Returns:
            The Tikhonov penalty for the given parameters.
        """
        penalty = 0
        for alt in range(0,self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt].copy().reshape(self.K.get_num_cols(), 1)  # Get only the column for alt
            penalty += alpha_alt.T.dot(self.K.dot(alpha_alt, index=alt)).item()
        penalty = pmle_lambda * penalty
        return penalty

    def tikhonov_penalty_gradient(self,
                                  alpha: np.ndarray,
                                  pmle_lambda: float
    ) -> np.ndarray:
        """Calculate the gradient of the Tikhonov penalty for the given parameters.

        Args:
            alpha: The vector of parameters.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.

        Returns:
            The gradient of the Tikhonov penalty for the given parameters.
        """
        return self.K.get_num_rows() * pmle_lambda * alpha
