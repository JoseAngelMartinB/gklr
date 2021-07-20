import sys
import time
from pympler import asizeof
from pympler.tracker import SummaryTracker
import numpy as np

from sklearn.metrics import pairwise
from sklearn.kernel_approximation import Nystroem

from gklr.estimation import Estimation
from gklr.calcs import Calcs


DTYPE = np.float64
VALID_PMLE_METHODS = [None, "Tikhonov"]
DEFAULT_NYSTROM_COMPRESION = 0.1

# Create a dictionary relating the kernel type parameter to the class from sklearn.gaussian_process.kernels that
# implements that kernel.
kernel_type_to_class = {"rbf": pairwise.rbf_kernel,
                        }

valid_kernel_list = kernel_type_to_class.keys()


def convert_size_bytes_to_human_readable(size_in_bytes):
   """ Convert the size from bytes to other units like KB, MB or GB"""
   if size_in_bytes < 1024:
       return (size_in_bytes, "Bytes")
   elif size_in_bytes < (1024*1024):
       return (np.round(size_in_bytes/1024, 2), "KB")
   elif size_in_bytes < (1024*1024*1024):
       return (np.round(size_in_bytes/(1024*1024), 2), "MB")
   else:
       return (np.round(size_in_bytes/(1024*1024*1024), 2), "GB")

def elapsed_time_to_str(elapsed_time_sec):
    if elapsed_time_sec > 60:
        return("{time:.2f} minutes".format(time=elapsed_time_sec/60))
    else:
        return("{time:.2f} seconds".format(time=elapsed_time_sec))



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
            kernel_params["nystrom"]
            del kernel_params["nystrom"]
        if "compresion" in kernel_params:
            del kernel_params["compresion"]

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


class KernelMatrix():
    def __init__(self, X, choice_column, obs_column, attributes, kernel_params, Z=None):
        # TODO: Check arguments
        # Create a new kernel based on the kernel type selected
        self._kernel_params = None
        self._kernel = None
        self._K = None
        self.nystrom = False
        self.nystrom_compresion = DEFAULT_NYSTROM_COMPRESION
        self.alternatives = None
        self.K_per_alternative = dict()
        self.alt_index = dict()
        self.n_cols = 0
        self.n_rows = 0
        self.choices = None
        self.choices_indices = None
        self.choices_matrix = None

        # Create the kernel matrix K
        self._create_kernel_matrix(X, choice_column, obs_column, attributes, kernel_params, Z)

    def _create_kernel_matrix(self, X, choice_column, obs_column, attributes, kernel_params, Z=None):
        # TODO: Check that attributescontains more than 1 alternative
        # TODO: Check that none alternative from attributes contains no attributes at all (giving a 0x0 matrix)
        self._kernel_params = kernel_params.copy()
        if "kernel" in kernel_params:
            kernel_type = kernel_params["kernel"]
            del kernel_params["kernel"]
        else:
            kernel_type = "rbf"
        if "nystrom" in kernel_params:
            self.nystrom = kernel_params["nystrom"]
            del kernel_params["nystrom"]
        if "compresion" in kernel_params:
            self.nystrom_compresion = kernel_params["compresion"]
            del kernel_params["compresion"]

        if self.nystrom == True:
            # TODO: Check that Z is None
            pass
        else:
            self._kernel = kernel_type_to_class[kernel_type]

        # If no reference dataframe Z is provided, then X will be the reference dataframe
        if Z is None:
            Z = X

        # Store the alternatives (classes) available
        self.alternatives = np.fromiter(attributes.keys(), dtype=int)

        # Initialize a dict K that contains the kernel matrix per each alternative
        self._K = dict()

        # Obtain the Kernel Matrix for each choice alternative
        index = 0
        for alt in self.alternatives:
            # Add the index of the alternative to `alt_index`
            self.alt_index[alt] = index

            # Obtain the list of attributes to be considered for alternative `alt`
            alt_attributes = attributes[alt]

            # Check if the kernel matrix for that alternative is already stored (same matrix)
            self.K_per_alternative[index] = index
            for prev_alt in self.alt_index.keys():
                if alt_attributes == attributes[prev_alt]:
                    self.K_per_alternative[index] = self.alt_index[prev_alt]
                    break

            if self.K_per_alternative[index] == index:
                # Obtain a submatrix X_alt and Z_alt from matrix X and Z, respectively,
                # with only the desired alternative `alt` and the selected attributes
                X_alt = X[alt_attributes]
                Z_alt = Z[alt_attributes]

                # Create the Kernel Matrix for alternative i
                if self.nystrom:
                    nystrom_components = int(X_alt.shape[0] * self.nystrom_compresion)
                    nystrom_kernel = Nystroem(kernel=kernel_type, n_components = nystrom_components, **kernel_params)
                    K_aux = nystrom_kernel.fit_transform(X_alt)
                else:
                    K_aux = self._kernel(Z_alt, X_alt, **kernel_params).astype(DTYPE)
                self._K[index] = K_aux

            index += 1

        # Store the number of columns and rows on the kernel matrix
        if self.n_rows == 0:
            self.n_rows = K_aux.shape[0]
        if self.n_cols == 0:
            if self.nystrom:
                # The matrix must be symmetric
                self.n_cols = self.n_rows
            else:
                self.n_cols = K_aux.shape[1]

        # Store the choices per observation
        self.choices = Z[choice_column]

    def get_num_cols(self):
        """Return the number of columns of the kernel matrix, which corresponds to the number of reference observations.
        """
        return self.n_cols

    def get_num_rows(self):
        """Return the number of rows of the kernel matrix, which corresponds to the number of observations.
        """
        return self.n_rows

    def get_alternatives(self):
        return self.alternatives

    def get_num_alternatives(self):
        return self.alternatives.shape[0]

    def get_choices(self):
        return self.choices.to_numpy()

    def get_choices_indices(self):
        if self.choices_indices is None:
            choice_indices = []
            for choice in self.choices.to_list():
                choice_indices.append(self.alt_index[choice])
            self.choices_indices = np.array(choice_indices)
        return self.choices_indices

    def get_choices_matrix(self):
        """Obtain a sparse matrix with one row per observation and one column per alternative. A cell Z_ij of the matrix
        takes value 1 if individual i choses alternative j; The cell contains 0 otherwise.
        """
        if self.choices_matrix is None:
            Z = np.zeros((self.get_num_rows(), self.get_num_alternatives()))
            Z[np.arange(len(Z)), self.get_choices_indices()] = 1
            self.choices_matrix = Z
        return self.choices_matrix

    def get_K(self, alt=None, index=None):
        """Returns the kernel matrix for all the alternatives, for alternative `alt`, or the matrix at index `index`.
        """
        if index is None:
            if alt is None:
                return self._K
            else:
                if alt in self.alt_index.keys():
                    return self._K[self.K_per_alternative[self.alt_index[alt]]]
                else:
                    raise ValueError(
                        "ERROR. Alternative `alt` = {alt} is not valid alternative. There is no kernel matrix "
                        "asociated with this alternative.".format(alt=alt))
        else:
            return self._K[self.K_per_alternative[index]]

    def dot(self, A, index=0):
        """Implements the dot product of the kernel matrix and numpy array A.
        """
        if self.nystrom:
            B = self.get_K(index=index).dot(self.get_K(index=index).T.dot(A))
        else:
            B = self.get_K(index=index).dot(A)
        return B

class KernelCalcs(Calcs):
    def __init__(self, K):
        super().__init__(K)

    def calc_probabilities(self, alpha):
        f = self.calc_f(alpha)
        Y = self.calc_Y(f)
        G, G_j = self.calc_G(Y)
        P = self.calc_P(Y, G, G_j)
        return P

    def log_likelihood(self, alpha, return_P=False):
        P = self.calc_probabilities(alpha)
        log_P = np.log(P)
        log_P = np.log(self.calc_probabilities(alpha))
        log_likelihood = np.sum(log_P[np.arange(len(log_P)), self.K.get_choices_indices()]) # TODO: .copy)  ??
        if return_P:
            return (log_likelihood, P)
        else:
            return log_likelihood

    def log_likelihood_and_gradient(self, alpha, pmle=None, pmle_lambda=0):
        # Log-likelihood
        (log_likelihood, P) = self.log_likelihood(alpha, return_P=True)

        # Gradient
        Z = self.K.get_choices_matrix()
        grad_penalization = 0
        if pmle is None:
            pass
        elif pmle == "Tikhonov":
            grad_penalization = self.tikhonov_penalty_gradient(alpha, pmle_lambda)
        else:
            raise ValueError("ERROR. {pmle} is not a valid value for the penalization method `pmle`.".format(
                pmle = pmle))

        H = grad_penalization + P - Z

        gradient = np.ndarray((self.K.get_num_rows(), 0), dtype=DTYPE)
        for alt in range(0,self.K.get_num_alternatives()):
            gradient_alt = self.K.dot(H[:, alt], index=alt)
            gradient_alt = (gradient_alt / H.shape[0]).reshape((self.K.get_num_rows(),1))
            gradient = np.concatenate((gradient, gradient_alt), axis=1)

        gradient = gradient.reshape(self.K.get_num_rows() * self.K.get_num_alternatives())

        return (log_likelihood, gradient)

    def calc_f(self, alpha):
        f = np.ndarray((self.K.get_num_rows(), 0), dtype=DTYPE)
        for alt in range(0,self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt].copy().reshape(self.K.get_num_cols(), 1)  # Get only the column for alt
            f_alt = self.K.dot(alpha_alt, index=alt)
            f = np.concatenate((f, f_alt), axis=1)
        return f

    def calc_G(self, Y):
        # Implementation for KLR
        G = np.sum(Y, axis=1).reshape((Y.shape[0], 1))
        # Compute G_j, the derivative of G with respecto to the variable Y_j
        G_j = np.ones_like(Y)
        return (G, G_j)

    def tikhonov_penalty(self, alpha, pmle_lambda):
        penalty = 0
        for alt in range(0,self.K.get_num_alternatives()):
            alpha_alt = alpha[:, alt].copy().reshape(self.K.get_num_cols(), 1)  # Get only the column for alt
            penalty += alpha_alt.T.dot(self.K.dot(alpha_alt, index=alt)).item()
        penalty = pmle_lambda * penalty
        return penalty

    def tikhonov_penalty_gradient(self, alpha, pmle_lambda):
        return self.K.get_num_rows() * pmle_lambda * alpha


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
