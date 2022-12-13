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

    def objective_function(self,
                           params: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Compute the objective function for the Kernel Logistic Regression 
        (KLR) model and its gradient.

        Args:
            params: The model parameters. Shape: (n_params,).

        Returns:
            A tuple with the value of the objective function and its gradient.
            The first element of the tuple is the value of the objective function
            and the second element is the gradient of the objective function with 
            respect to the model parameters with shape: (num_rows_kernel_matrix * num_alternatives,)
        """
        # Convert params to alfas and reshape them as a column vector
        alpha = params.reshape(self.alpha_shape)

        # Compute the log-likelihood and gradient
        ll, gradient = self.calcs.log_likelihood_and_gradient(alpha, self.pmle, self.pmle_lambda)

        # Compute the penalty function
        penalty = 0
        if self.pmle is None:
            pass
        elif self.pmle == "Tikhonov":
            penalty = self.calcs.tikhonov_penalty(alpha, self.pmle_lambda)
        else:
            msg = f"'pmle' = {self.pmle} is not a valid value for the penalization."
            logger_error(msg)
            raise ValueError(msg)

        self.history["loss"].append(-ll + penalty)
        self.history["gradient"].append(gradient)

        if self.verbose >= 2:
            print(f"Current objective function: {-ll+penalty:,.4f}", end = "\r")
            sys.stdout.flush()
        return (-ll + penalty, gradient)

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
