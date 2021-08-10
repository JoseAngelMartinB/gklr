import numpy as np
from scipy.optimize import minimize

class Estimation():
    def __init__(self, calcs, pmle, pmle_lambda, method, verbose):
        self.calcs = calcs
        self.pmle = pmle
        self.pmle_lambda = pmle_lambda
        self.method = method
        self.verbose = verbose

    def objective_function(self):
        pass

    def minimize(self, params, loss_tol=1e-06, options=None):
        # Default
        gradient_tol = 1e-06
        maxiter = 1000

        options = {'gtol': gradient_tol,
                   "maxiter": maxiter,
                   "maxls": 30}
        res = minimize(self.objective_function, params, method=self.method, jac=True, tol=loss_tol, options=options)
        results = {
            "fun": res.fun, # Final value of the objective function
            "params": res.x, # The solution array
            "success": res.success, # A boolean flag indicating if the optimizer exited successfully
            "message": res.message, # A string that describes the cause of the termination
        }
        return results
