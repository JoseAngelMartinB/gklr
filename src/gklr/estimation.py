"""GKLR estimation module."""
from abc import ABC, abstractmethod
from optparse import Option
from typing import Optional, Any, Dict, List, Union

import numpy as np
from scipy.optimize import minimize

from .calcs import Calcs
from .optimizer import Optimizer
from .kernel_utils import *
from .logger import *

class Estimation(ABC):
    """Base Estimation class object."""

    def __init__(self,
                 calcs: Calcs,
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
        if method not in SCIPY_OPTIMIZATION_METHODS + CUSTOM_OPTIMIZATION_METHODS:
            msg = (f"'method' = {method} is not a valid optimization method.\n"
                   f"Valid methods are: {SCIPY_OPTIMIZATION_METHODS + CUSTOM_OPTIMIZATION_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)

        self.calcs = calcs
        self.pmle = pmle
        self.pmle_lambda = pmle_lambda
        self.method = method
        self.verbose = verbose
        self.history = {
            'loss': [], # Loss function per iteration
            'time': None, # Time per iteration (Not implemented yet for SCIPY_OPTIMIZATION_METHODS)
            }
        self.n_samples = calcs.K.get_num_samples()

    @abstractmethod
    def objective_function(self):
        return

    @abstractmethod
    def gradient(self):
        return

    @abstractmethod
    def objective_function_with_gradient(self):
        return

    def minimize(self,
                 params: np.ndarray,
                 loss_tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimize the objective function.
        
        Args:
            params: The initial values of the model parameters. Shape: (n_params,).
            loss_tol: The tolerance for the loss function. Default: 1e-06.
            options: A dict with advance options for the optimization method. 
                Default: None.
            
        Returns:
            A dict with the results of the optimization.
        """
        # Default parameters for the optimization method
        gradient_tol = 1e-06
        maxiter = 1000

        if options is None:
            options = {}
        options = dict(options)
        options.setdefault('gtol', gradient_tol)
        options.setdefault('maxiter', maxiter)
        options.setdefault('maxls', 30)
        if self.method == "SGD" or self.method == "momentumSGD" or self.method == "adam":
            options.setdefault('n_samples', self.n_samples)

        if self.method in SCIPY_OPTIMIZATION_METHODS:
            # Use the scipy.optimize.minimize function
            jac = self.gradient
            res = minimize(self.objective_function, params, method=self.method, jac=jac, tol=loss_tol, options=options)
        elif self.method in CUSTOM_OPTIMIZATION_METHODS:
            # Use the custom optimization function
            optimizer = Optimizer()
            jac = self.gradient
            res = optimizer.minimize(self.objective_function, params, method=self.method, jac=jac, tol=loss_tol, options=options)
            # Override default history values because possible minibatches store only the disaggretated loss
            self.history['loss'] = res["history"]["loss"]
            self.history['time'] = res["history"]["time"]
        else:
            msg = f"Error: The optimization method '{self.method}' is not valid."
            logger_error(msg)
            raise ValueError(msg)
        results = {
            "fun": res["fun"], # Final value of the objective function
            "params": res["x"], # The solution array
            "success": res["success"], # A boolean flag indicating if the optimizer exited successfully
            "message": res["message"], # A string that describes the cause of the termination
        }
        return results
