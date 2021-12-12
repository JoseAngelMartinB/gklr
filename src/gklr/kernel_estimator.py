import sys

from gklr.kernel_utils import *
from gklr.estimation import Estimation

class KernelEstimator(Estimation):
    def __init__(self, calcs, pmle, pmle_lambda, method, verbose):
        if pmle not in VALID_PMLE_METHODS:
            raise ValueError("ERROR. {pmle} is not a valid value for the penalization method `pmle`. Valid methods "
                             "are: {valid_methods}".format(pmle=pmle, valid_methods=VALID_PMLE_METHODS))

        super().__init__(calcs, pmle, pmle_lambda, method, verbose)
        self.alpha_shape = (calcs.K.get_num_cols(), calcs.K.get_num_alternatives())

    def objective_function(self, params):
        #time_ini = time.time_ns()  # DEBUG
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
            raise ValueError("ERROR. {pmle} is not a valid value for the penalization method `pmle`.".format(
                pmle = self.pmle))

        if self.verbose >= 2:
            print("Current objective function: {fun:,.4f}".format(fun=-ll+penalty), end = "\r")
            sys.stdout.flush()
        #print(params, end="\r") #DEBUG:
        #print((time.time_ns() - time_ini) / (10 ** 9))  # convert to floating-point seconds) # DEBUG
        return (-ll + penalty, gradient)

    def minimize(self, params):
        #DEBUG: tracker = SummaryTracker()
        results = super().minimize(params)
        # Convert params to alfas and reshape them as a column vector
        results["alpha"] = results["params"].reshape(self.alpha_shape)
        #DEBUG: tracker.print_diff()
        return results