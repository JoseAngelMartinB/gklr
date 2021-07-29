import sys
import time
from pympler import asizeof
from pympler.tracker import SummaryTracker
import numpy as np

from gklr.kernel_utils import *
from gklr.kernel_estimator import KernelEstimator
from gklr.kernel_calcs import KernelCalcs
from gklr.kernel_matrix import KernelMatrix


class KernelModel:
    """Main class for GKLR models.
    """
    def __init__(self, model_params=None):
        self._X = None
        self.choice_column = None
        self.obs_column = None
        self.attributes = None
        self.kernel_params = None
        self._Z = None
        self._K = None
        self._K_test = None
        self._alpha = None
        self.alpha_shape = None
        self.n_parameters = 0
        self.results = None

        if model_params == None:
            self._model_params = None
        else:
            # TODO: Check parameters
            self._model_params = model_params

    def _create_kernel_matrix(self, X, choice_column, obs_column, attributes, kernel_params, Z=None, train=True):
        success = 1 # TODO: check when success is 0
        # TODO: ensure_columns_are_in_dataframe
        # TODO: ensure_valid_variables_passed_to_kernel_matrix

        if train:
            self._K = KernelMatrix(X, choice_column, obs_column, attributes, kernel_params, Z)
            self.n_parameters = self._K.get_num_cols() * self._K.get_num_alternatives() # One alpha vector per alternative
            self.alpha_shape = (self._K.get_num_cols(), self._K.get_num_alternatives())
        else:
            self._K_test = KernelMatrix(X, choice_column, obs_column, attributes, kernel_params, Z)
        return success

    def get_kernel(self, dataset="train"):
        if dataset == "train":
            return self._K
        elif dataset == "test":
            return self._K_test
        elif dataset == "both":
            return (self._K, self._K_test)
        else:
            raise ValueError("dataset must be a value in: ['train', 'test', 'both']")

    def clear_kernel(self, dataset="train"):
        if dataset == "train":
            self._K = None
        elif dataset == "test":
            self._K_test = None
            self._Z = None
        elif dataset == "both":
            self._K = None
            self._K_test = None
            self._X = None
            self._Z = None
        else:
            raise ValueError("dataset must be a value in: ['train', 'test', 'both']")
        return None

    def set_kernel_train(self, X, choice_column, obs_column, attributes, kernel_params, verbose=1):
        self.clear_kernel(dataset="both")
        start_time = time.time()
        success = self._create_kernel_matrix(X, choice_column, obs_column, attributes, kernel_params.copy(), train=True)
        elapsed_time_sec = time.time() - start_time

        if success == 0:
            print("ERROR. The kernel matrix for the train set have not been created.")
            sys.stdout.flush()
            return None
        else:
            self._X = X
            self.choice_column = choice_column
            self.obs_column = obs_column
            self.attributes = attributes
            self.kernel_params = kernel_params

        if verbose >= 1:
            elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)
            K_size, K_size_u = convert_size_bytes_to_human_readable(asizeof.asizeof(self._K))
            print("The kernel matrix for the train set have been correctly created in {elapsed_time}. "
                  "Size of the matrix object: {sizeK} {unit}".format(elapsed_time=elapsed_time_str, sizeK=K_size,
                                                                  unit = K_size_u))
            sys.stdout.flush()
        return None

    def set_kernel_test(self, Z, choice_column=None, obs_column=None, attributes=None, kernel_params=None, verbose=1):
        if self._K is None:
            print("ERROR. First you must compute the kernel for the train dataset using set_kernel_train().")
            return None

        self.clear_kernel(dataset="test")

        # Set default values for the input parameters
        choice_column = self.choice_column if choice_column is None else choice_column
        obs_column = self.obs_column if obs_column is None else obs_column
        attributes = self.attributes if attributes is None else attributes
        kernel_params = self.kernel_params.copy() if kernel_params is None else kernel_params

        # Nystrom method is not allowed for the test kernel
        if "nystrom" in kernel_params:
            del kernel_params["nystrom"]
        if "compression" in kernel_params:
            del kernel_params["compression"]

        start_time = time.time()
        success = self._create_kernel_matrix(self._X, choice_column, obs_column, attributes, kernel_params, Z=Z,
                                             train=False)
        elapsed_time_sec = time.time() - start_time

        if success == 0:
            print("ERROR. The kernel matrix for the test set have not been created.")
            sys.stdout.flush()
            return None
        else:
            self._Z = Z

        if verbose >= 1:
            elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)
            K_size, K_size_u = convert_size_bytes_to_human_readable(asizeof.asizeof(self._K_test))
            print("The kernel matrix for the test set have been correctly created in {elapsed_time}. "
                  "Size of the matrix object: {sizeK} {unit}".format(elapsed_time=elapsed_time_str, sizeK=K_size,
                                                                     unit=K_size_u))
            sys.stdout.flush()
        return None

    def fit(self, init_parms = None, pmle="Tikhonov", pmle_lambda=0, method="L-BFGS-B", verbose=1):
        if self._K is None:
            print("ERROR. First you must compute the kernel for the train dataset using set_kernel_train().")
            return None

        if init_parms is None:
            init_parms = np.zeros(self.alpha_shape, dtype=DTYPE)
        else:
            pass # TODO: check that there are self.n_parameters and then make a cast to self.alpha_shape

        # Create the Calcs instance
        calcs = KernelCalcs(K=self._K)

        # Create the estimator instance
        estimator = KernelEstimator(calcs=calcs, pmle=pmle, pmle_lambda=pmle_lambda, method=method, verbose=verbose)

        # Log-likelihood at zero
        alpha_at_0 = np.zeros(self.alpha_shape, dtype=DTYPE)
        log_likelihood_at_zero = calcs.log_likelihood(alpha_at_0)

        # Initial log-likelihood
        initial_log_likelihood = calcs.log_likelihood(init_parms)

        if verbose >= 1:
            print("The estimation is going to start...\n"
                  "Log-likelihood at zero: {ll_zero:,.4f}\n"
                  "Initial log-likelihood: {i_ll:,.4f}".format(ll_zero=log_likelihood_at_zero, i_ll=initial_log_likelihood))
            sys.stdout.flush()
        if verbose >= 2:
            print("Number of parameters to be estimated: {n_parameters:,d}".format(n_parameters=self.n_parameters))
            sys.stdout.flush()

        # Perform the estimation
        start_time = time.time()
        self.results = estimator.minimize(init_parms.reshape(self.n_parameters))
        elapsed_time_sec = time.time() - start_time
        elapsed_time_str = elapsed_time_to_str(elapsed_time_sec)

        final_log_likelihood = calcs.log_likelihood(self.results["alpha"])
        mcfadden_r2 = 1 - final_log_likelihood / log_likelihood_at_zero  # TODO: Implement a method to compute metrics

        # Store post-estimation information
        self.results["final_log_likelihood"] = final_log_likelihood
        self.results["elapsed_time"] = elapsed_time_sec
        self.results["mcfadden_r2"] = mcfadden_r2
        self.results["pmle"] = pmle
        self.results["pmle_lamda"] = pmle_lambda
        self.results["method"] = method

        if verbose >= 1:
            print("-------------------------------------------------------------------------\n"
                  "The kernel model has been estimated. Elapsed time: {elapsed_time}.\n"
                  "Final log-likelihood value: {final_log_likelihood:,.4f}\n"
                  "McFadden R^2: {r2:.4f}".format(elapsed_time=elapsed_time_str,
                                                  final_log_likelihood=final_log_likelihood,
                                                  r2 = mcfadden_r2))
            sys.stdout.flush()

        return None

    def predict_proba(self, train=False):
        """Predict class probabilities for the train or test kernel.
        """
        if train:
            # Create the Calcs instance
            calcs = KernelCalcs(K=self._K)
        else:
            if self._K_test is None:
                print("ERROR. First you must compute the kernel for the test dataset using set_kernel_test().")
                return None
            # Create the Calcs instance
            calcs = KernelCalcs(K=self._K_test)

        proba = calcs.calc_probabilities(self.results["alpha"])
        return proba

    def predict_log_proba(self, train=False):
        """Predict the natural logarithm of the class probabilities for the train or test kernel.
        """
        proba = self.predict_proba(train)
        return np.log(proba)

    def predict(self, train=False):
        """Predict class for the train or test kernel.
        """
        proba = self.predict_proba(train)
        encoded_labels = np.argmax(proba, axis=1)
        return self._K.alternatives.take(encoded_labels)

    def score(self):
        """Predict the mean accuracy on the test kernel.
        """
        if self._K_test is None:
            print("ERROR. First you must compute the kernel for the test dataset using set_kernel_test().")
            return None

        y_true = self._Z[self.choice_column]
        y_predict = self.predict()
        score = np.average(y_true == y_predict)
        return score
