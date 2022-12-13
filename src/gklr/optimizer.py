"""GKLR optimizer module."""
from typing import Optional, Any, Dict, List, Union, Callable

import numpy as np
from scipy.optimize import OptimizeResult

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
                 x0: np.ndarray,
                 args: tuple = (),
                 method: str = "SGD",
                 jac: Optional[Union[Callable, bool]] = None,
                 hess: Optional[Callable] = None,
                 tol: float = 1e-06,
                 options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the specified optimization method.

        Args:
            fun: The objective function to be minimized.
                ``fun(x, *args) -> float``,
                where ``x`` is the input vector and ``args`` are the additional
                arguments of the objective function.
            x0: The initial guess of the parameters.
            args: Additional arguments passed to the objective function.
            method: The optimization method. Default: "SGD".
            jac: The gradient of the objective function. Default: None.
            hess: The Hessian of the objective function. Default: None.
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

        if options is None:
            # Use the default options
            options = {}

        if tol is not None:
            options = dict(options)
            if method == 'SGD':
                options.setdefault('gtol', tol)

        if callable(jac):
            pass
        elif jac is True:
            # fun returns the objective function and the gradient
            fun = MemoizeJac(fun)
            jac = fun.derivative
        elif jac is None:
            jac = None
        else:
            # Default option if jac is not understood
            jac = None

        # TODO: Hessians are not implemented yet

        if method == "SGD":
            # Use the stochastic gradient descent method
            res = self._minimize_sgd(fun, x0, jac, args, **options)
        else:
            msg = (f"'method' = {method} is not a valid optimization method.\n"
                   f"Valid methods are: {CUSTOM_OPTIMIZATION_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)


        return res # TODO: return the results


    def _minimize_sgd(self,
                      fun: Callable,
                      x0: np.ndarray,
                      jac: Optional[Callable] = None,
                      args: tuple = (),
                      learning_rate: float = 1e-03,
                      gtol: float = 1e-06, 
                      startiter: int = 0,
                      maxiter: int = 1000,
                      momentum: float = 0.0,
                      **kwards,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the stochastic gradient descent method.
        """

        # Checking errors
        if not callable(fun):
            m = "The objective function must be callable."
            logger_error(m)
            raise ValueError(m)
        if learning_rate <= 0:
            m = "The learning rate must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if gtol <= 0:
            m = "The tolerance must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if startiter < 0:
            m = "The iteration number must be non-negative."
            logger_error(m)
            raise ValueError(m)
        if maxiter <= 0:
            m = "The maximum number of iterations must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if jac is None:
            # TODO: Implement the gradient-free optimization method using 2-point approximation
            m = "The gradient of the objective function must be provided."
            logger_error(m)
            raise ValueError(m)
        if momentum < 0 or momentum > 1:
            m = "The momentum must be in the range [0, 1]."
            logger_error(m)
            raise ValueError(m)

        n, = x0.shape
        g = np.zeros((n,), np.float64)
        velocity = np.zeros((n,), np.float64)
        message = "Optimization terminated successfully."
        success = True

        x = x0
        i = 0
        for i in range(startiter, startiter + maxiter):
            g = jac(x)
            velocity = momentum * velocity - learning_rate * g
            diff = velocity 
            if np.all(np.abs(diff) <= gtol):
                break
            x = x + diff
        i += 1

        if i >= maxiter:
            message = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            success = False

        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, 
            success=success, message=message)



class MemoizeJac:
    """ Decorator that caches the return values of a function returning `(fun, grad)`
        each time it is called. """

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None

    def _compute_if_needed(self, x, *args):
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            fg = self.fun(x, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        """ returns the the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac