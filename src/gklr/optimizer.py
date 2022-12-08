"""GKLR optimizer module."""
from typing import Optional, Any, Dict, List, Union, Callable

import numpy as np

from .kernel_utils import *
from .logger import *

class Optimizer():
    """Optimizer class object."""

    def __init__(self) -> None:
        """Constructor.
        """
        return

    def minimize(self,
                 fun: Callable,
                 params: np.ndarray,
                 method: str = "SGD",
                 tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the specified optimization method.

        Args:
            fun: The objective function to be minimized.
                ``fun(x, *args) -> float``,
                where ``x`` is the input vector and ``args`` are the additional
                arguments of the objective function.
            params: The initial guess of the parameters.
            method: The optimization method. Default: "SGD".
            tol: The tolerance for the termination. Default: 1e-06.
            options: A dictionary of solver options. Default: None.

        Returns:
            A dictionary containing the result of the optimization procedure:
                fun: The value of the objective function at the solution.
                x: A 1-D ndarray containing the solution.
                success: A boolean indicating whether the optimization converged.
                message: A string describing the cause of the termination.
        """
        if method is None:
            # Use the default method
            method = "SGD"

        if method == "SGD":
            # Use the stochastic gradient descent method
            res = self._minimize_sgd(fun, params, tol=tol, options=options)
        else:
            msg = (f"'method' = {method} is not a valid optimization method.\n"
                   f"Valid methods are: {CUSTOM_OPTIMIZATION_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)


        return res # TODO: return the results


    def _minimize_sgd(self,
                      fun: Callable,
                      params: np.ndarray,
                      tol: float = 1e-06, 
                      options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the stochastic gradient descent method.
        """
        
        # TODO: implement the stochastic gradient descent method
        raise NotImplementedError("The stochastic gradient descent method is not implemented yet.")

        pass