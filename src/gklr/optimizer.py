"""GKLR optimizer module."""
from typing import Optional, Any, Dict, List, Union, Callable, Tuple

import sys
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult

from .kernel_utils import *
from .logger import *


class LearningRateScheduler:
    """Implements different learning rate scheduling methods."""

    def __init__(self, 
                 lr_scheduler: Optional[str] = None,
                 lr_decay_rate: float = 1,
                 lr_decay_step: int = 100,
    ) -> None:
        """Initialize the learning rate scheduler.

        Args:
            lr_scheduler: The method for the learning rate decay. Default: None.
            lr_decay_rate: The learning rate decay rate. Default: 1.
            lr_decay_step: The learning rate decay step for the step decay method.
            Default: 100.
        """
        self.lr_scheduler = lr_scheduler
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        if lr_scheduler is not None and \
           lr_scheduler not in ["time-based","exponential","step"]:
            m = "Unknown learning rate scheduling method: {}".format(lr_scheduler)
            logger_error(m)
            raise ValueError(m)

    def __call__(self,
                 learning_rate0: float,
                 epoch: int,
    ) -> float:
        """Update the learning rate.
        
        Args:
            learning_rate0: Initial learning rate.
            epoch: Current epoch (iteration).
            
        Returns:
            Updated learning rate.
        """
        if self.lr_scheduler is None:
            return learning_rate0
        elif self.lr_scheduler == "time-based":
            return self._update_lr_time_based(learning_rate0, epoch, self.lr_decay_rate)
        elif self.lr_scheduler == "exponential":
            return self._update_lr_exponential(learning_rate0, epoch, self.lr_decay_rate)
        elif self.lr_scheduler == "step":
            return self._update_lr_step(learning_rate0, epoch, self.lr_decay_rate, self.lr_decay_step)
        else:
            m = "Unknown learning rate scheduling method: {}".format(self.lr_scheduler)
            logger_error(m)
            raise ValueError(m)

    def _update_lr_time_based(self,
                              learning_rate0: float,
                              epoch: int,
                              decay_rate: float,
    ) -> float:
        """Update the learning rate using the time-based decay method.

        Args:
            learning_rate0: Initial learning rate.
            epoch: Current epoch (iteration).
            decay_rate: Decay rate.

        Returns:
            Updated learning rate.
        """
        learning_rate = learning_rate0/(1+decay_rate*epoch)
        return learning_rate

    def _update_lr_exponential(self,
                               learning_rate0: float,
                               epoch: int,
                               decay_rate: float,
    ) -> float:
        """Update the learning rate using the exponential decay method.

        Args:
            learning_rate0: Initial learning rate.
            epoch: Current epoch (iteration).
            decay_rate: Decay rate.

        Returns:
            Updated learning rate.
        """
        learning_rate = learning_rate0*np.exp(-decay_rate*epoch)
        return learning_rate

    def _update_lr_step(self,
                        learning_rate0: float,
                        epoch: int,
                        decay_rate: float,
                        decay_step: int,
    ) -> float:
        """Update the learning rate using the step decay method.

        Args:
            learning_rate0: Initial learning rate.
            epoch: Current epoch (iteration).
            decay_rate: Decay rate.
            decay_steps: Decay steps.

        Returns:
            Updated learning rate.
        """
        learning_rate = learning_rate0/(1+decay_rate*np.floor(epoch/decay_step))
        return learning_rate


class AcceleratedLinearSearch:
    """Class for the accelerated linear search algorithm."""

    def __init__(self,
                 gamma: float = 1.1,
                 theta: float = 0.5,
                 max_alpha: float = 1.5,
                 n_epochs: int = 10,
    ) -> None:
        """Initialize the accelerated linear search algorithm.

        Args:
            gamma: The gamma parameter. Default: 1.1.
            theta: The theta parameter. Default: 0.5.
            max_alpha: The maximum alpha value. Default: 1.5.
            n_epochs: Number of epochs in the main algorithm to perform one step
                of the accelerated linear search. Default: 10.
        """
        self.gamma = gamma
        self.theta = theta
        self.max_alpha = max_alpha
        self.alpha_t = 0
        self.n_epochs = n_epochs
        self.epoch = 0 # Current epoch
        self.w_t = None # Value of the parameters at the previous iteration

    def initialize(self,
                   y_t: np.ndarray,
    ) -> None:
        """Initialize the accelerated linear search algorithm.
        
        Parameters:
            y_t: The value of the parameters at the current iteration.
        """
        self.alpha_t = self.max_alpha/self.gamma # Initialize the alpha value
        self.epoch = 0
        self.w_t = y_t # Value of the parameters at the previous iteration

    def update_params(self,
                      fun: Callable,
                      y_t: np.ndarray,
                      *args,
    ) -> np.ndarray:
        """Execute the accelerated linear search algorithm and update the parameters.

        Args:
            fun: The objective function to be minimized.
                ``fun(x, *args) -> float``,
                where ``x`` is the input vector and ``args`` are the additional
                arguments of the objective function.
            y_t: The value of the parameters at the current iteration.
            *args: Additional arguments of the objective function.

        Returns:
            The new value of the weights.
        """
        if self.w_t is None:
            # The accelerated linear search algorithm has not been initialized
            self.initialize(y_t)
            m = ("The accelerated linear search algorithm has not been "
                 "previously initialized. The algorithm has been initialized "
                 "with the current value of the parameters.")
            logger_warning(m)
            return y_t

        self.epoch += 1
        if self.epoch == self.n_epochs:
            # Execute the accelerated linear search algorithm
            self.epoch = 0
            
            # Compute a search direction
            d_t = y_t - self.w_t

            # Compute the function estimates
            F0_t = fun(self.w_t, *args)

            # Compute the step size
            mu_t = self.alpha_t
            delta = self.theta * np.linalg.norm(d_t, ord=2)
            if F0_t - fun(self.w_t + mu_t*d_t, *args) >= mu_t*delta:
                # Increase the step size while the Armijo condition is satisfied
                armijo_satisfied = True
                while armijo_satisfied and (self.gamma*mu_t <= self.max_alpha):
                    mu_t = self.gamma*mu_t
                    if F0_t - fun(self.w_t + mu_t*d_t, *args) < mu_t*delta:
                        armijo_satisfied = False
            else:
                # Otherwise, decrease the step size until the Armijo condition 
                # is satisfied
                armijo_satisfied = False
                while not armijo_satisfied and (1 <= mu_t/self.gamma):
                    mu_t = mu_t/self.gamma
                    if F0_t - fun(self.w_t + mu_t*d_t, *args) >= mu_t*delta:
                        armijo_satisfied = True
            
            # Update the parameters
            self.alpha_t = mu_t
            next_w_t = self.w_t + self.alpha_t*d_t
            self.w_t = y_t
            return next_w_t
        else:
            # The accelerated linear search algorithm is not executed
            return y_t


class MemoizeJac:
    """Decorator that caches the return values of a function returning `(fun, grad)`
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
        """Returns the the function value."""
        self._compute_if_needed(x, *args)
        return self._value

    def derivative(self, x, *args):
        self._compute_if_needed(x, *args)
        return self.jac


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
            options.setdefault('gtol', tol)
            # TODO: Currently, the tolerance is not used by any optimization method

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

        # Initialize the learning rate scheduler
        if "learning_rate_scheduler" in options:
            learning_rate_scheduler = options["learning_rate_scheduler"]
            if learning_rate_scheduler is not None and not callable(learning_rate_scheduler):
                msg = (f"'learning_rate_scheduler' = {learning_rate_scheduler} is not a valid function.\n"
                        f"Valid functions are: callable.")
                logger_error(msg)
                raise ValueError(msg)
        elif "lr_scheduler" in options:
            # Load the user-defined parameters
            lr_scheduler = options["lr_scheduler"]
            options.pop("lr_scheduler")
            decay_opts = {}
            if "lr_decay_rate" in options:
                decay_opts["lr_decay_rate"] = options["lr_decay_rate"]
                options.pop("lr_decay_rate")
            if "lr_decay_step" in options:
                decay_opts["lr_decay_step"] = options["lr_decay_step"]
                options.pop("lr_decay_step")
            learning_rate_scheduler = LearningRateScheduler(lr_scheduler, **decay_opts)
            options["learning_rate_scheduler"] = learning_rate_scheduler

        # Set parameters for the AcceleratedLinearSearch
        if "accelerated_linear_search" in options:
            if options["accelerated_linear_search"] is True:
                # Load the user-defined parameters
                als_options = {}
                if "als_gamma" in options:
                    als_options["gamma"] = options["als_gamma"]
                    options.pop("als_gamma")
                if "als_theta" in options:
                    als_options["theta"] = options["als_theta"]
                    options.pop("als_theta")
                if "als_max_alpha" in options:
                    als_options["max_alpha"] = options["als_max_alpha"]
                    options.pop("als_max_alpha")
                if "als_n_epochs" in options:
                    als_options["n_epochs"] = options["als_n_epochs"]
                    options.pop("als_n_epochs")
                accelerated_linear_search = AcceleratedLinearSearch(**als_options)
                options["accelerated_linear_search"] = accelerated_linear_search
            else:
                options["accelerated_linear_search"] = None

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
                                learning_rate_scheduler: Optional[Callable] = None,
                                accelerated_linear_search: Optional[AcceleratedLinearSearch] = None,
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
                Default: 1e-08.
            learning_rate_scheduler: A function that computes the learning rate
                at each iteration. Default: None.
            accelerated_linear_search: An instance of the AcceleratedLinearSearch
                class. If None, the accelerated linear search is not used.
                Default: None.
            maxiter: The maximum number of iterations or epochs. Default: 1000.
            print_every: The number of iterations to print the loss. If 0,
                the loss is not computed. If -1, the loss is computed at each 
                iteration but not printed. If -2, the loss and time per iteration
                are computed but not printed. Default: 0. 
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
        learning_rate0 = learning_rate # Initial learning rate
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
        if accelerated_linear_search is not None:
            # Initialize the accelerated linear search
            accelerated_linear_search.initialize(x0)
        history = {
            "loss": [],
            "time": [],
        }
        message = "Optimization terminated successfully."
        success = True

        # Optimization loop
        x = x0
        epoch = 0
        for epoch in range(num_epochs):
            epoch_init_time = 0
            if print_every == -2:
                # Store the time at the beginning of the epoch
                epoch_init_time = perf_counter()
            
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
                if print_every < 0 or (print_every > 0 and epoch%print_every == 0):
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

            # Linear search acceleration
            if accelerated_linear_search is not None:
                x = accelerated_linear_search.update_params(fun, x, *args)

            # Update the learning rate
            if learning_rate_scheduler is not None:
                learning_rate = learning_rate_scheduler(learning_rate0, epoch)

            # Print the average loss of the mini-batches if it is required
            if print_every < 0 or (print_every > 0 and epoch%print_every == 0):
                history["loss"].append(epoch_loss)
                if print_every == -2:
                    # Store the time consumed by the epoch
                    history["time"].append(perf_counter() - epoch_init_time)
                if print_every > 0:
                    # Print the loss at the current epoch
                    print(f"\t* Epoch: {epoch}/{num_epochs} - Avg. loss: {epoch_loss:.4f}")
                sys.stdout.flush()

        epoch += 1
        if epoch >= num_epochs:
            message = 'STOP: TOTAL NO. of ITERATIONS REACHED LIMIT'
            success = True

        return OptimizeResult(x=x, fun=fun(x,*args), jac=g, nit=epoch, nfev=epoch, 
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