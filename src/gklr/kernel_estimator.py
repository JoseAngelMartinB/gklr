"""GKLR kernel_estimator module."""
from cmath import log
from typing import Optional, Tuple, Any, Dict

from asyncio.log import logger
import sys

import numpy as np

from .kernel_calcs import KernelCalcs
from .logger import *
from .kernel_utils import *
from .estimation import Estimation

class KernelEstimator(Estimation):
    """Estimation object for the Kernel Logistic Regression (KLR) model."""

    def __init__(self,
                 calcs: KernelCalcs,
                 pmle: Optional[str] = None,
                 pmle_lambda: float = 0.0,
                 method: str = "L-BFGS-B",
                 verbose: int = 1,
    ) -> None:
        """Constructor.

        Args:
            calcs: Calcs object.
            pmle: Indicates the penalization method for the penalized maximum
                likelihood estimation. If 'None' a maximum likelihood estimation
                without penalization is performed. Default: None.
            pmle_lambda: The value of the regularization parameter for the PMLE
                method. Default: 0.0.
            method: The optimization method. Default: "L-BFGS-B".
            verbose: Indicates the level of verbosity of the function. If 0, no
                output will be printed. If 1, basic information about the
                estimation procedure will be printed. If 2, the information
                about each iteration will be printed. Default: 1.
        """
        if pmle not in VALID_PMLE_METHODS:
            msg = (f"'pmle' = {pmle} is not a valid value for the penalization"
                   f" method. Valid methods are: {VALID_PMLE_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)

        super().__init__(calcs, pmle, pmle_lambda, method, verbose)
        self.calcs = calcs
        self.alpha_shape = (calcs.K.get_num_cols(), calcs.K.get_num_alternatives())
        self.P_cache = None # Cache for the matrix of probabilities P
        self.prev_params = None # Previous parameters used in the objective function
        self.prev_indices = None # Previous indices used in the objective function

    def objective_function(self,
                           params: np.ndarray,
                           indices: Optional[np.ndarray] = None
    ) -> float:
        """Compute the objective function for the Kernel Logistic Regression 
        (KLR) model and its gradient.

        Args:
            params: The model parameters. Shape: (n_params,).
            indices: The indices of the samples to be used in the computation of
                the objective function. If 'None' all the samples will be used.
                Default: None.
        Returns:
            A tuple with the value of the objective function and its gradient.
            The first element of the tuple is the value of the objective function
            and the second element is the gradient of the objective function with 
            respect to the model parameters with shape: (num_rows_kernel_matrix * num_alternatives,)
        """
        # Convert params to alfas and reshape them as a column vector
        alpha = params.reshape(self.alpha_shape)

        if self.prev_params is None or not np.array_equal(params, self.prev_params) or \
            (indices is not None and self.prev_indices is None) or \
            (indices is None and self.prev_indices is not None) or \
            (indices is not None and self.prev_indices is not None and \
            not np.array_equal(indices, self.prev_indices)):
            # Compute the matrix of probabilities P and store it in the cache
            P = self.calcs.calc_probabilities(alpha, indices=indices)
            self.P_cache = P
            self.prev_params = params
            self.prev_indices = indices
        else:
            # Reuse the cached matrix of probabilities P
            P = self.P_cache

        # Compute the log-likelihood
        ll = self.calcs.log_likelihood(alpha, P=P, pmle=self.pmle, pmle_lambda=self.pmle_lambda, indices=indices)
        self.history["loss"].append(-ll)

        if self.verbose >= 2:
            print(f"Current objective function: {-ll:,.4f}", end = "\r")
            sys.stdout.flush()
        return (-ll)

    def gradient(self,
                 params: np.ndarray,
                 indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the gradient of the objective function for the Kernel Logistic
        Regression (KLR) model.
        
        This function is used by the optimization methods that do not require
        the computation of the objective function. If the objective function is
        also required, it is more efficient to use the 'objective_function'
        method, setting the 'return_gradient' argument to 'True'.

        Args:
            params: The model parameters. Shape: (n_params,).
            indices: The indices of the samples to be used in the computation of
                the the gradient. If 'None' all the samples will be used.
                Default: None.
        
        Returns:
            The gradient of the objective function with respect to the model
            parameters with shape: (num_rows_kernel_matrix * num_alternatives,).
        """
        # Convert params to alfas and reshape them as a column vector
        alpha = params.reshape(self.alpha_shape)

        if self.prev_params is None or not np.array_equal(params, self.prev_params) or \
            (indices is not None and self.prev_indices is None) or \
            (indices is None and self.prev_indices is not None) or \
            (indices is not None and self.prev_indices is not None and \
            not np.array_equal(indices, self.prev_indices)):
            # Compute the matrix of probabilities P and store it in the cache
            P = self.calcs.calc_probabilities(alpha, indices=indices)
            self.P_cache = P
            self.prev_params = params
            self.prev_indices = indices
        else:
            # Reuse the cached matrix of probabilities P
            P = self.P_cache

        # Compute the log-likelihood and gradient
        gradient = self.calcs.gradient(alpha, P=P, pmle=self.pmle, pmle_lambda=self.pmle_lambda, indices=indices)
        return gradient

    def objective_function_with_gradient(self,
                                         params: np.ndarray,
                                         indices: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """Compute the objective function for the Kernel Logistic Regression 
        (KLR) model and its gradient.

        Args:
            params: The model parameters. Shape: (n_params,).
            indices: The indices of the samples to be used in the computation of
                the objective function. If 'None' all the samples will be used.
                Default: None.
        Returns:
            A tuple with the value of the objective function and its gradient.
            The first element of the tuple is the value of the objective function
            and the second element is the gradient of the objective function with 
            respect to the model parameters with shape: (num_rows_kernel_matrix * num_alternatives,)
        """
        # Compute the log-likelihood and gradient
        obj = self.objective_function(params, indices=indices)
        gradient = self.gradient(params, indices=indices)
        return (obj, gradient)


    def minimize(self,
                 params: np.ndarray,
                 loss_tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
                 **kargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Minimize the objective function.

        Args:
            params: The initial values of the model parameters. Shape: (n_params,).
            loss_tol: The tolerance for the loss function. Default: 1e-06.
            options: A dict with advance options for the optimization method. 
                Default: None.
            **kargs: Additional arguments for the minimization function.

        Returns:
            A dict with the results of the optimization.
        """
        results = super().minimize(params, loss_tol, options, **kargs)
        # Convert params to alpha np vector and reshape them as a column vector
        results["alpha"] = results["params"].reshape(self.alpha_shape)
        return results
