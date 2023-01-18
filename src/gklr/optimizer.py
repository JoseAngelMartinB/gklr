"""GKLR optimizer module."""
from typing import Optional, Any, Dict, List, Union, Callable, Tuple

import sys

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
            elif method == 'momentumSGD':
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
            # Use the mini-batch stochastic gradient descent method
            res = self.minimize_mini_batch_sgd(fun, x0, optimizer="SGD", jac=jac, args=args, **options)
        elif method == "momentumSGD":
            # Use the mini-batch stochastic gradient descent method with momentum
            res = self.minimize_mini_batch_sgd(fun, x0, optimizer="momentumSGD", jac=jac, args=args, **options)
        elif method == "adam":
            # Use the mini-batch Adam method
            res = self.minimize_mini_batch_sgd(fun, x0, optimizer="adam", jac=jac, args=args, **options)
        else:
            msg = (f"'method' = {method} is not a valid optimization method.\n"
                   f"Valid methods are: {CUSTOM_OPTIMIZATION_METHODS}.")
            logger_error(msg)
            raise ValueError(msg)

        return res

    def minimize_mini_batch_sgd(self,
                                fun: Callable,
                                x0: np.ndarray,
                                jac: Optional[Callable] = None,
                                optimizer: str = "SGD",
                                args: tuple = (),
                                learning_rate: float = 1e-03,
                                mini_batch_size: Optional[int] = None,
                                n_samples: int = 0,
                                beta: float = 0.9,
                                beta1: float = 0.9,
                                beta2: float = 0.999,
                                epsilon: float = 1e-08,
                                gtol: float = 1e-06, 
                                maxiter: int = 1000, # Number of epochs
                                print_every: int = 0,
                                seed: int = 0,
                                **kwards,
    ) -> Dict[str, Any]:
        """Minimize the objective function using the mini-batch stochastic 
        gradient descent method.

        Args:
            fun: The objective function to be minimized.
                ``fun(x, *args) -> float``,
                where ``x`` is the input vector and ``args`` are the additional
                arguments of the objective function.
            x0: The initial guess of the parameters.
            jac: The gradient of the objective function. Default: None.
            optimizer: The variant of the mini-batch stochastic gradient descent
                method to be used. Default: "SGD".
            args: Additional arguments passed to the objective function.
            learning_rate: The learning rate. Default: 1e-03.
            mini_batch_size: The mini-batch size. Default: None.
            n_samples: The number of samples in the dataset. Default: 0.
            beta: The momentum parameter. Default: 0.9.
            beta1: The exponential decay rate for the first moment estimates
                (gradients) in the Adam method. Default: 0.9.
            beta2: The exponential decay rate for the second moment estimates 
                (squared gradients) in the Adam method. Default: 0.999.
            epsilon: A small constant for numerical stability in the Adam method.
            gtol: The tolerance for the termination. Default: 1e-06.
            maxiter: The maximum number of iterations or epochs. Default: 1000.
            print_every: The number of iterations to print the loss. Default: 0. 
            seed: The seed for the random number generator. Default: 0.
            **kwards: Additional arguments passed to the objective function.

        Returns:
            A dictionary containing the result of the optimization procedure:
                fun: The value of the objective function at the solution.
                x: A 1-D ndarray containing the solution.
                jac: The gradient of the objective function at the solution.
                nit: The number of iterations.
                nfev: The number of function evaluations.
                success: A boolean indicating whether the optimization converged.
                message: A string describing the cause of the termination.
                history: A dictionary containing the loss history.
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
        if mini_batch_size is not None and mini_batch_size <= 0:
            m = "The mini-batch size must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if n_samples <= 0:
            m = ("The number of samples in the dataset must be greater than zero"
                 " and corresponds with number of rows in the dataset.")
            logger_error(m)
            raise ValueError(m)
        if mini_batch_size is not None and mini_batch_size > n_samples:
            m = "The mini-batch size must be less than or equal to the number of samples in the dataset."
            logger_error(m)
            raise ValueError(m)
        if gtol <= 0:
            m = "The tolerance must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if maxiter <= 0:
            m = "The maximum number of iterations (epochs) must be greater than zero."
            logger_error(m)
            raise ValueError(m)
        if jac is None:
            # TODO: Implement the gradient-free optimization method using 2-point approximation
            m = "The gradient of the objective function must be provided."
            logger_error(m)
            raise ValueError(m)


        # Initialize parameters
        num_epochs = maxiter
        n, = x0.shape
        g = np.zeros((n,), np.float64)
        v = np.ndarray(0, np.float64)
        s = np.ndarray(0, np.float64)
        t = 0 # Adam counter
        if optimizer == "SGD":
            pass
        elif optimizer == "momentumSGD":
            # Initialize velocity
            v = np.zeros((n,), np.float64)
        elif optimizer == "adam":
            # Initialize velocity (v) and the square of the gradient (s)
            v = np.zeros((n,), np.float64)
            s = np.zeros((n,), np.float64)
        history = {
            "loss": [],
        }
        message = "Optimization terminated successfully."
        success = True

        # Optimization loop
        x = x0
        i = 0
        for i in range(num_epochs):
            if mini_batch_size is None:
                # Use the entire dataset as the mini-batch (batch gradient descent)
                minibatches = [None]
            else:
                # Define the random mini-batches. Increment the seed to reshuffle differently at each epoch
                seed += 1
                minibatches = self._random_mini_batch(n_samples, mini_batch_size, seed=seed)

            diff = np.zeros((n,), np.float64)
            epoch_loss = 0

            for minibatch in minibatches:
                # Compute the loss of the mini-batch if it is required
                if print_every > 0 and i % print_every == 0:
                    epoch_loss += fun(x, minibatch, *args)

                # Compute the gradient
                g = jac(x, minibatch, *args)
                
                # Update the parameters
                if optimizer == "SGD":
                    x = self._update_parameters_SGD(x, g, learning_rate)
                elif optimizer == "momentumSGD":
                    x, v = self._update_parameters_momentumSGD(x, g, v, learning_rate, beta)
                elif optimizer == "adam":
                    t = t + 1 # Adam counter
                    x, v, s = self._update_parameters_adam(x, g, v, s, t, learning_rate, beta1, beta2, epsilon)
                else:
                    m = f"Optimizer '{optimizer}' is not supported."
                    logger_error(m)
                    raise ValueError(m)

                diff = - learning_rate * g
                x = x + diff

            # Print the average loss of the mini-batches if it is required
            if print_every > 0 and i % print_every == 0:
                history["loss"].append(epoch_loss)
                print(f"\t* Epoch: {i}/{num_epochs} - Avg. loss: {epoch_loss:.4f}")
                sys.stdout.flush()
                
            if np.all(np.abs(diff) <= gtol):
                # Convergence
                message = "Optimization terminated successfully. Gradient tolerance reached."
                break

        i += 1
        if i >= num_epochs:
            message = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            success = False

        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, 
            success=success, message=message, history=history)

    def _update_parameters_SGD(self,
                               x: np.ndarray,
                               g: np.ndarray,
                               learning_rate: float,
     ) -> np.ndarray:
          """Update the parameters using the mini-batch SGD method."""
          return x - learning_rate * g

    def _update_parameters_momentumSGD(self,
                                       x: np.ndarray,
                                       g: np.ndarray,
                                       v: np.ndarray,
                                       learning_rate: float,
                                       beta: float,
     ) -> Tuple[np.ndarray, np.ndarray]:
        """Update the parameters using the momentum mini-batch SGD method."""
        # Update the velocity
        v = beta*v + (1-beta)*g
        # Update the parameters
        x = x - learning_rate*v
        return x, v

    def _update_parameters_adam(self,
                                x: np.ndarray,
                                g: np.ndarray,
                                v: np.ndarray,
                                s: np.ndarray,
                                t: int,
                                learning_rate: float,
                                beta1: float,
                                beta2: float,
                                epsilon: float,
     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the parameters using the mini-batch Adam method."""
        # Update the velocity
        v = beta1*v + (1-beta1)*g
        # Compute bias-corrected first moment estimate
        v_hat = v / (1-np.power(beta1,t))
        # Update the squared gradients
        s = beta2*s + (1-beta2)*np.power(g,2)
        # Compute bias-corrected second raw moment estimate
        s_hat = s/(1-np.power(beta2,t))
        # Update the parameters
        x = x - learning_rate*v_hat/(np.sqrt(s_hat)+epsilon)
        return x, v, s

    def _random_mini_batch(self,
                           n_samples: int,
                           mini_batch_size: int,
                           seed: int = 0,
    ) -> List[np.ndarray]:
        """
        Generate a list of random minibatches for the indices [0, ..., n_samples - 1]
        """
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        mini_batches = []
        for i in range(0, n_samples, mini_batch_size):
            mini_batch = indices[i:i + mini_batch_size]
            # Sort the indices of each mini-batch to improve memory access time
            mini_batch.sort()
            mini_batches.append(mini_batch)
        return mini_batches


class MemoizeJac:
    """ Decorator that caches the return values of a function returning `(fun, grad)`
        each time it is called. """

    def __init__(self, fun):
        self.fun = fun
        self.jac = None
        self._value = None
        self.x = None
        self.minibatch = None

    def _compute_if_needed(self, x, minibatch = None,  *args):
        # Check if the function value has already been computed for the given x
        if not np.all(x == self.x) or self._value is None or self.jac is None:
            self.x = np.asarray(x).copy()
            if minibatch is not None:
                self.minibatch = np.asarray(minibatch).copy()
            else:
                self.minibatch = None
            fg = self.fun(x, minibatch, *args)
            self.jac = fg[1]
            self._value = fg[0]
        
        # Check if the mini-batches are the same as previous ones
        if not np.all(minibatch == self.minibatch):
            self.minibatch = np.asarray(minibatch).copy()
            fg = self.fun(x, minibatch, *args)
            self.jac = fg[1]
            self._value = fg[0]

    def __call__(self, x, *args):
        """ returns the the function value """
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac