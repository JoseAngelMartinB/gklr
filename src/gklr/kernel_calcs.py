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

    def calc_probabilities(self, 
                           alpha: np.ndarray,
                           indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the probabilities for each alternative.

        Obtain the probabilities for each alternative for each row of the
        dataset.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            indices: The indices of the rows of the dataset for which the
                probabilities are calculated. If None, the probabilities are
                calculated for all rows of the dataset. Default: None.

        Returns:
            A matrix of probabilities for each alternative for each row of the
                dataset. Each column corresponds to an alternative and each row
                to a row of the dataset. The sum of the probabilities for each
                row is 1. Shape: (n_samples, num_alternatives).
        """
        f = self.calc_f(alpha, indices=indices)
        Y = self.calc_Y(f)
        G, G_j = self.calc_G(Y)
        P = self.calc_P(Y, G, G_j)
        return P

    def log_likelihood(self,
                       alpha: np.ndarray,
                       P: Optional[np.ndarray] = None,
                       choice_indices: Optional[np.ndarray] = None,
                       pmle: Optional[str] = None,
                       pmle_lambda: float = 0,
                       indices: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate the log-likelihood of the KLR model for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            P: The matrix of probabilities of each alternative for each row of 
                the dataset. If None, the probabilities are calculated.
                Shape: (n_samples, num_alternatives). Default: None.
            choice_indices: The indices of the chosen alternatives for each row
                of the dataset. If None, the indices are obtained from the
                KernelMatrix object. Shape: (n_samples,). Default: None.
            pmle: It specifies the type of penalization for performing a penalized
                maximum likelihood estimation.  Default: None.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
                 Default: 0.
            indices: The indices of the rows of the dataset for which the
                log-likelihood is calculated. If None, the log-likelihood is
                calculated for all rows of the dataset. Default: None.

        Returns:
            The log-likelihood of the KLR model for the given parameters.
        """
        if indices is None:
            num_rows = self.K.get_num_rows()
        else:
            num_rows = indices.shape[0]
        if P is None:
            P = self.calc_probabilities(alpha, indices=indices)
        else:
            if P.shape != (num_rows, self.K.get_num_alternatives()):
                m = (f"P has {P.shape} dimensions, but it should have "
                    f" dimensions: ({num_rows}, {self.K.get_num_alternatives()}).")
                logger_error(m)
                raise ValueError(m)
        if choice_indices is None:
            choice_indices = self.K.get_choices_indices()
            if indices is not None:
                indices = indices.tolist()
                choice_indices = choice_indices[indices]
        else:
            if len(choice_indices) != P.shape[0]:
                m = (f"choice_indices has {len(choice_indices)} elements, but P"
                    " has {P.shape[0]} rows.")
                logger_error(m)
                raise ValueError(m)

        # Compute the log-likelihood from the matrix of probabilities
        log_P = np.log(P)
        log_likelihood = np.sum(log_P[np.arange(len(log_P)), choice_indices]) 

        # Compute the penalty function
        penalty = 0
        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            penalty = self.tikhonov_penalty(alpha, pmle_lambda)
        else:
            msg = f"'pmle' = {pmle} is not a valid value for the penalization."
            logger_error(msg)
            raise ValueError(msg)

        return log_likelihood + penalty

    def gradient(self,
                 alpha: np.ndarray,
                 P: Optional[np.ndarray] = None,
                 pmle: Optional[str] = None,
                 pmle_lambda: float = 0,
                 indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the gradient of the log-likelihood function of the KLR model 
        for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            pmle: It specifies the type of penalization for performing a penalized
                maximum likelihood estimation.  Default: None.
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
                 Default: 0.
            P: The matrix of probabilities of each alternative for each row of 
                the dataset. If None, the probabilities are calculated.
                Shape: (n_samples, num_alternatives). Default: None.
            indices: The indices of the rows of the dataset for which the
                log-likelihood is calculated. If None, the log-likelihood is
                calculated for all rows of the dataset. Default: None.

        Returns:
            The gradient of the log-likelihood function of the KLR model for the
            given parameters. Shape: (num_rows_kernel_matrix * num_alternatives, ).
        """
        if indices is None:
            num_rows = self.K.get_num_rows()
        else:
            num_rows = indices.shape[0]
        if P is None:
            P = self.calc_probabilities(alpha, indices=indices)
        else:
            if P.shape != (num_rows, self.K.get_num_alternatives()):
                m = (f"P has {P.shape} dimensions, but it should have "
                    f" dimensions: ({num_rows}, {self.K.get_num_alternatives()}).")
                logger_error(m)
                raise ValueError(m)
        
        # Compute the gradient of the log-likelihood function
        Z = self.K.get_choices_matrix()
        if indices is not None:
            Z = Z[indices, :]
        grad_penalization = 0
        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            grad_penalization = self.tikhonov_penalty_gradient(alpha, pmle_lambda, indices=indices)
        else:
            msg = f"ERROR. {pmle} is not a valid value for the penalization method `pmle`."
            logger_error(msg)
            raise ValueError(msg)
        H = grad_penalization + P - Z

        n_alts = self.K.get_num_alternatives()
        gradient = np.zeros((self.K.get_num_cols(), n_alts), dtype=DEFAULT_DTYPE)
        for alt in range(0,n_alts):
            gradient_alt = self.K.dot(H[:, alt], K_index=alt, col_indices=indices)
            gradient_alt = (gradient_alt / H.shape[0]).reshape((self.K.get_num_cols(),))
            gradient[:, alt] = gradient_alt
        gradient = gradient.reshape(self.K.get_num_cols() * self.K.get_num_alternatives())
        return gradient

    def calc_f(self, 
              alpha: np.ndarray, 
              indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the value of utility function for each alternative for each row
        of the dataset.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            indices: The indices of the rows of the dataset for which the utility
                function is calculated. If None, all the rows are used. Default: None.

        Returns:
            A matrix where each row corresponds to the utility of each alternative
                for each row of the dataset. Shape: (n_samples, num_alternatives).
        """
        num_rows = self.K.get_num_rows()
        if indices is not None:
            if np.max(indices) >= self.K.get_num_rows() or np.min(indices) < 0:
                msg = "Some of the indices provided to compute utility function are out of range."
                logger_error(msg)
                raise ValueError(msg)
            else:
                num_rows = indices.shape[0]

        n_alts = self.K.get_num_alternatives()
        f = np.zeros((num_rows, n_alts), dtype=DEFAULT_DTYPE)
        for alt in range(0,n_alts):
            alpha_alt = alpha[:, alt].reshape(self.K.get_num_cols(),1)  # Get only the column for alt
            f_alt = self.K.dot(alpha_alt, K_index=alt, row_indices=indices)
            f[:, alt] = f_alt.reshape((num_rows,))  # Store the result in the corresponding column
        return f

    def calc_G(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the generating function `G` of a Generalized Extreme Value
            (GEV) model and its derivative. For KLR model, the generating function
            is the sum of the utilities of the alternatives for each row of the
            dataset.

        Args:
            Y: The auxiliary matrix `Y` that contains the exponentiated values of the
                matrix `f`. Shape: (n_samples, num_alternatives).

        Returns:
            A tuple with the auxiliary matrix `G` and its derivative.
                The auxiliary matrix `G` is a numpy array of shape: (n_samples, 1)
                and its derivative `G_j` is a numpy array of shape: (n_samples, num_alternatives).
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
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.

        Returns:
            The Tikhonov penalty for the given parameters.
        """
        penalty = 0
        for alt in range(0,self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt].reshape(self.K.get_num_cols(), 1)  # Get only the column for alt
            penalty += alpha_alt.T.dot(self.K.dot(alpha_alt, K_index=alt)).item()
        penalty = 0.5 * pmle_lambda * penalty
        return penalty # TODO: Check if this is correct or if a subset of the rows should be used for alpha

    def tikhonov_penalty_gradient(self,
                                  alpha: np.ndarray,
                                  pmle_lambda: float,
                                  indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calculate the gradient of the Tikhonov penalty for the given parameters.

        Args:
            alpha: The vector of parameters. Shape: (num_cols_kernel_matrix, num_alternatives).
            pmle_lambda: The lambda parameter for the penalized maximum likelihood.
            indices: The indices of the rows of the dataset for which the gradient

        Returns:
            The gradient of the Tikhonov penalty for the given parameters.
                If indices is None, the shape is (num_cols_kernel_matrix, num_alternatives),
                otherwise, the shape is (len(indices), num_alternatives).
        """
        if indices is not None:
            alpha = alpha[indices.tolist(), :] # Get only the rows for the indices
        return alpha.shape[0] * pmle_lambda * alpha
