"""GKLR kernel_matrix module."""
from typing import Optional, Any, Dict, List, Union

import math
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import svd, eigvalsh
from sklearn.cluster import KMeans, MiniBatchKMeans

from .logger import *
from .config import Config
from .kernel_utils import *

__all__ = ['KernelMatrix']

class KernelMatrix():
    """Class to store the kernel matrix and its associated data."""
    
    def __init__(self,
                 X: pd.DataFrame,
                 choice_column: str,
                 attributes: Dict[int, List[str]],
                 config: Config,
                 Z: Optional[pd.DataFrame] = None,
    ) -> None:
        """Constructor.

        Args:
            X: Train dataset stored in a pandas DataFrame. Shape: (n_samples, n_features).
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            config: A Config object that contains the hyperparameters of the GKLR model.
            Z: Test dataset stored in a pandas DataFrame. Shape: (n_samples, n_features).
                Default: None
        """
        # TODO: Check arguments
        # Create a new kernel based on the kernel type selected
        self._config = config
        self._kernel = None
        self._K = None
        self.nystrom = False
        self.nystrom_sampling = "uniform"
        self.nystrom_compression = DEFAULT_NYSTROM_COMPRESSION
        self.alternatives = []
        self.n_alternatives = 0
        self.K_per_alternative = dict()
        self.alt_to_index = dict() # Links each alternative with an index
        self.n_cols = 0
        self.n_rows = 0
        self.choices = None
        self.choices_indices = None
        self.choices_matrix = None
        self.n_samples = 0

        # Create the kernel matrix K
        self._init_kernel_matrix(X, choice_column, attributes, Z)

    def _init_kernel_matrix(self,
                            X: pd.DataFrame,
                            choice_column: str,
                            attributes: Dict[int, List[str]],
                            Z: Optional[pd.DataFrame] = None,
    ) -> None:
        """Construct and store the kernel matrix K.

        The kernel matrix K is constructed using the training dataset `X` and the
        test dataset `Z`. The kernel matrix is stored in the attribute `self._K`.
        The kernel matrix dimensions are (n_rows, n_cols), where n_rows is the
        number of rows of the training dataset `X` and n_cols is the number of
        rows of the test dataset `Z`. If `Z` is not provided, then `X` is used
        as the test dataset and n_cols = n_rows.
        If Nyström is used, the dimension of the kernel matrix is reduced to
        (n_rows, nystrom_components), where nystrom_components is the number of
        components (or landmarks) used for the Nyström approximation.

        Args:
            X: Train dataset stored in a pandas DataFrame. Shape: (n_samples, n_features).
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            Z: Test dataset stored in a pandas DataFrame. Shape: (n_samples, n_features).
                Default: None
        """
        # TODO: Check that attributes contains more than 1 alternative
        # TODO: Check that none alternative from attributes contains no attributes at all (giving a 0x0 matrix)

        # Initialize the kernel parameters
        kernel_type = self._config["kernel"]
        if kernel_type is None:
            msg = "Hyperparameter 'kernel' is not specified. Set to 'rbf'."
            logger_warning(msg)
            kernel_type = "rbf"

        if Z is None:
            # Initialize training kernel only parameters
            self.nystrom = self._config["nystrom"]
            self.nystrom_compression = self._config["compression"]
            self.nystrom_sampling = self._config["nystrom_sampling"]
            # If no reference dataframe Z is provided, then X will be the reference dataframe
            Z = X
        else:
            self.nystrom = False

        # TODO: Implement n_jobs (parallelization)

        # Store the number of samples in dataset X
        self.n_samples = X.shape[0]

        # Store the alternatives (classes) available
        self.alternatives = list(np.fromiter(attributes.keys(), dtype=int))
        self.n_alternatives = len(self.alternatives)

        # Initialize a dict K that contains the kernel matrix per each alternative
        self._K = dict()

        # Obtain the Kernel Matrix for each choice alternative
        index = 0 
        for alt in self.alternatives:
            # Add the index of the alternative to `alt_to_index`
            self.alt_to_index[alt] = index

            # Obtain the list of attributes to be considered for alternative `alt`
            alt_attributes = attributes[alt]

            # Assign to the alternative a default kernel matrix index
            self.K_per_alternative[index] = index
            # Check if the kernel matrix for that alternative was already stored (same matrix)
            for prev_alt in self.alt_to_index.keys():
                if alt_attributes == attributes[prev_alt]:
                    self.K_per_alternative[index] = self.alt_to_index[prev_alt]
                    break

            if self.K_per_alternative[index] == index:
                # Obtain a submatrix X_alt and Z_alt from matrix X and Z, respectively,
                # with only the desired alternative `alt` and the selected attributes
                X_alt = X[alt_attributes]
                Z_alt = Z[alt_attributes]

                # Create the Kernel Matrix for alternative i
                if self.nystrom:
                    if self.nystrom_compression <= 1 and self.nystrom_compression > 0:
                        nystrom_components = int(X_alt.shape[0] * self.nystrom_compression)
                    elif self.nystrom_compression > 1:
                        nystrom_components = int(self.nystrom_compression)
                    else:
                        msg = "'compresion' hyperparameter must be a positive number."
                        logger_error(msg)
                        raise ValueError(msg)
                    nystrom_kernel = Nystroem(kernel=kernel_type, n_components = nystrom_components,
                        n_jobs=self._config["n_jobs"], sampling=self.nystrom_sampling, **self._config["kernel_params"])
                    K_aux = nystrom_kernel.fit_transform(X_alt)
                else:
                    self._kernel = kernel_type_to_class[kernel_type]
                    K_aux = self._kernel(Z_alt, X_alt, **self._config["kernel_params"]).astype(DEFAULT_DTYPE)
                self._K[index] = K_aux

            index += 1

        # Store the number of columns and rows on the kernel matrix
        self.n_rows = self._K[0].shape[0]
        if self.nystrom:
            # The matrix must be symmetric
            self.n_cols = self.n_rows
        else:
            self.n_cols = self._K[0].shape[1]

        # Store the choices per observation
        self.choices = Z[choice_column]

    def get_num_cols(self) -> int:
        """Return the number of columns of the kernel matrix.
        
        Returns:
            Number of columns of the kernel matrix, which corresponds to the 
                number of reference observations.
        """
        return self.n_cols

    def get_num_rows(self) -> int:
        """Return the number of rows of the kernel matrix.

        Returns:
            Number of rows of the kernel matrix, which corresponds to the
                number of observations.
        """
        return self.n_rows

    def get_num_samples(self) -> int:
        """Return the number of observations in the dataset.
        
        Returns:
            Number of observations in the dataset.
        """
        return self.n_samples

    def get_alternatives(self) -> np.ndarray:
        """Return the available alternatives.

        Returns:
            A numpy array with the available alternatives. Shape: (n_alternatives,).
        """
        return np.array(self.alternatives)

    def get_num_alternatives(self) -> int:
        """Return the number of available alternatives.
        
        Returns:
            Number of available alternatives."""
        return self.n_alternatives

    def get_choices(self) -> np.ndarray:
        """Return the choices per observation.

        Returns:
            A numpy array with the choices per observation. Shape: (n_samples,).
        """
        if self.choices is None:
            msg = "Kernel matrix not initialized."
            logger_error(msg)
            raise RuntimeError(msg)
        return self.choices.to_numpy()

    def get_choices_indices(self) -> np.ndarray:
        """Return the choices per observation as alternative indices.

        Returns:
            A numpy array with the choices per observation as alternative indices.
            Shape: (n_samples,).
        """
        if self.choices is None:
            msg = "Kernel matrix not initialized."
            logger_error(msg)
            raise RuntimeError(msg)
        if self.choices_indices is None:
            choice_indices = []
            for choice in self.choices.to_list():
                choice_indices.append(self.alt_to_index[choice])
            self.choices_indices = np.array(choice_indices)
        return self.choices_indices

    def get_choices_matrix(self) -> np.ndarray:
        """Return the choices per observation as a matrix.

        Obtain a sparse matrix with one row per observation and one column per alternative. A cell Z_ij of the matrix
        takes value 1 if individual i choses alternative j; The cell contains 0 otherwise.

        Returns:
            A numpy array with the choices per observation as a matrix.
                Shape: (n_samples, n_alternatives).
        """
        if self.choices_matrix is None:
            Z = np.zeros((self.get_num_rows(), self.get_num_alternatives()))
            Z[np.arange(len(Z)), self.get_choices_indices()] = 1
            self.choices_matrix = Z
        return self.choices_matrix

    def get_K(self,
              alt: Optional[int] = None,
              index: Optional[int] = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """Returns the kernel matrix for all the alternatives, for alternative `alt`, or the matrix at index `index`.

        Args:
            alt: Alternative for which the kernel matrix to be returned.
            index: Index of the kernel matrix to be returned.

        Returns:
            The kernel matrix for all the alternatives, for alternative `alt`, or the matrix at index `index`.
        """
        if self._K is None:
            msg = "Kernel matrix not initialized."
            logger_error(msg)
            raise RuntimeError(msg)
            
        if index is None and alt is None:
            return self._K
        elif index is None and alt is not None:
            if alt in self.alt_to_index.keys():
                return self._K[self.K_per_alternative[self.alt_to_index[alt]]]
            else:
                msg = (f"Alternative 'alt' = {alt} is not valid alternative. There is no kernel matrix "
                        "asociated with this alternative.")
                logger_error(msg)
                raise ValueError(msg)
        elif index is not None and alt is None:
            if index < self.n_alternatives:
                return self._K[self.K_per_alternative[index]]
            else:
                msg = (f"'index' = {index} is not valid index. There is no kernel matrix "
                        "with this index.")
                logger_error(msg)
                raise ValueError(msg)
        else:
            msg = (f"The arguments 'alt' and 'index' cannot be used at the same time.")
            logger_error(msg)
            raise ValueError(msg)

    def dot(self, 
            A: np.ndarray,
            K_index: int = 0,
            row_indices: Optional[np.ndarray] = None,
            col_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Implements the dot product of the kernel matrix and numpy array A.

        Implements the matrix multiplication K ∙ A, where K is the kernel matrix 
        and A is a numpy array given as argument.

        Args:
            A: Numpy array to be multiplied by the kernel matrix. 
                Shape: (num_cols_kernel_matrix, •)
            K_index: Index of the kernel matrix to be used.
            row_indices: Indices of the rows of the kernel matrix to be used in
                the dot product. If None, all the rows are used. Default: None.
            col_indices: Indices of the columns of the kernel matrix to be used
                in the dot product. If None, all the columns are used. Default: None.

        Returns:
            The dot product of the kernel matrix and `A`.
                Shape: (num_rows_kernel_matrix, •)
        """
        K = self.get_K(index=K_index)
        assert isinstance(K, np.ndarray)
        if self.nystrom:
            # Compute the dot product using the Nyström approximation of the kernel matrix.
            if row_indices is not None:
                # A subset of the rows in the kernel matrix is used.
                row_indices = row_indices.tolist()
                B = K[row_indices, :].dot(K.T.dot(A))
            elif col_indices is not None:
                # A subset of the columns in the kernel matrix is used.
                n_cols = col_indices.shape[0]
                col_indices = col_indices.tolist()
                if A.shape[0] != n_cols:
                    msg = (f"Error in K.dot(): "
                            f"The number of columns in the kernel matrix ({n_cols}) "
                            f"does not match the number of rows in the array A ({A.shape[0]}).")
                    logger_error(msg)
                    raise ValueError(msg)
                B = K.dot(K.T[:, col_indices].dot(A))
            else:
                # Default case: All rows and columns are used.
                B = K.dot(K.T.dot(A))
        else:
            # Compute the dot product using the full kernel matrix.
            if row_indices is not None:
                row_indices = row_indices.tolist()
                K = K[row_indices, :]
            if col_indices is not None:
                col_indices = col_indices.tolist()
                K = K[:, col_indices]
            B = K.dot(A)
        return B


class Nystroem():
    """Nyström approximation of a kernel matrix.

    This class implements the Nyström approximation of a kernel matrix. The 
    Nyström approximation is a method to approximate a kernel matrix K using a
    smaller matrix, which allows to reduce the computational cost of the
    kernel methods.
    This class is based on the implementation of the Nyström method in the
    scikit-learn library.
    """
    def __init__(self,
                 kernel = "rbf",
                 *,
                 gamma = None,
                 coef0 = None,
                 degree = None,
                 kernel_params = None,
                 n_components = 100,
                 sampling = "uniform",
                 ridge_leverage_lambda = 1,
                 random_state = None,
                 n_jobs = None,
    ):
        """Constructor.

        Args:
            kernel: String with the name of the kernel to be used. Default = "rbf".
            gamma: Kernel coefficient for "rbf", "poly" and "sigmoid". Default = None.
            coef0: Independent term in kernel function. It is only significant in "poly" and "sigmoid". Default = None.
            degree: Degree of the polynomial kernel function ("poly"). Default = None.
            kernel_params: Additional parameters (keyword arguments) for kernel function passed as dictionary. Default = None.
            n_components: Number of components of the Nyström approximation. Default = 100.
            sampling: String with the sampling method to be used for obtaining the components for the Nyström approximation.
                Default = "uniform".
            ridge_leverage_lambda: Lambda parameter for the `DAC-ridge-leverage` sampling method. Default = 1.
            random_state: Random state to be used. Default = None.
            n_jobs: Number of jobs to be used. Default = None.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.sampling = sampling
        self.ridge_leverage_lambda = ridge_leverage_lambda

    def fit_transform(self,
                      X,
                      y = None,
                      **fit_params,
    ):
        """Fit the Nyström approximation to the data and obtain the Nyström approximation of the kernel matrix.

        Args:
            X: Data to be used for fitting the Nyström approximation. array-like of shape: (n_samples, n_features)
            y: Target values. Default = None.
            fit_params: Additional parameters to be passed to the kernel function.

        Returns:
            The Nyström approximation of the kernel matrix.
        """
        return self.fit(X, **fit_params).transform(X)

    def fit(self,
            X,
            y = None,
    ):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Args:
            X: Data to be used for fitting the Nyström approximation. array-like of shape: (n_samples, n_features)
            y: Target values. Default = None.
        """
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            n_components = n_samples
            msg = ("n_components > n_samples. This is not possible.\n"
                   "n_components was set to n_samples, which results"
                   " in inefficient evaluation of the full kernel.")
            logger_warning(msg)
            
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)

        if self.sampling == "uniform":
            inds = rnd.permutation(n_samples)
            basis_inds = inds[:n_components]
            basis = X[basis_inds]
        elif self.sampling == "kmeans":
            kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=rnd, batch_size=n_components*5, max_iter=100)
            kmeans.fit_predict(X)
            basis = kmeans.cluster_centers_
        elif self.sampling == "DAC-ridge-leverage":
            approx_leverage_scores = self.DAC_ridge_leverage(X, self.ridge_leverage_lambda, n_components)
            # Sample n_components elements from data proportionally to these scores
            p = approx_leverage_scores/np.sum(approx_leverage_scores)  
            selected = np.random.choice(X.shape[0], size=n_components, replace=False, p=p)  
            basis = X[selected]
        elif self.sampling == "recursive-ridge-leverage":
            leverage_indices = self.recursive_ridge_leverage(X, n_components)
            basis = X[leverage_indices]
        else:
            msg = "{self.sampling} is not a valid sampling strategy for Nyström method."
            logger_error(msg)
            raise ValueError(msg)

        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        filter_params=True,
                                        n_jobs=self.n_jobs,
                                        **self._get_kernel_params())

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Args:
            X: Data to transform. array-like of shape: (n_samples, n_features)

        Returns:
            Transformed data. ndarray of shape: (n_samples, n_components)
        """
        X = check_array(X, accept_sparse='csr')
        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    filter_params=True,
                                    n_jobs=self.n_jobs,
                                    **kernel_params)
        return np.dot(embedded, np.transpose(self.normalization_))

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        return params

    def DAC_ridge_leverage(self,
                           X,
                           n_components,
                           lambda_,
    ):
        """DAC ridge-leverage algorithm for Nyström approximation.

        This function computes an approximation of the ridge leverage score, using a divide and conquer strategy.
        Reference: Farah Cherfaoui, Hachem Kadri, Liva Ralaivola.  Scalable ridge Leverage score sampling for the 
        Nyström method. ICASSP, May 2022, Singapour, Singapore.
        Based on the implementation of the authors: https://github.com/DAC-paper/Divide_And_Conquer_leverage_score_approximation 

        Args:
            X: Numpy array of size (n, d) where n is the number of data and d number of features.
            sample_size: Size of sub-matrix.
            lambda_: Regularisation term.
        """

        if lambda_ < 0:
            msg = "'lambda_' parameter of DAC_ridge_leverage have to be positive."
            logger_error(msg)
            raise ValueError(msg)

        n = X.shape[0]
        ind = np.arange(n)
        np.random.shuffle(ind)
        approximated_ls = np.zeros((n))

        for l in range(0, math.ceil(n/n_components)):
            # Sample a subset of data
            true_sample_size = min(n_components, n - l*n_components)
            temp_ind = ind[l*n_components: l*n_components + true_sample_size]
            
            # compute the kernel matrix using the subset of selected data
            K_S = pairwise_kernels(X[temp_ind], X[temp_ind],
                                   metric=self.kernel,
                                   filter_params=True,
                                   n_jobs=self.n_jobs,
                                   **self._get_kernel_params())

            # compute the approximated leverage score by inverting the small matrix
            approximated_ls[temp_ind] = np.sum(K_S * np.linalg.inv(K_S + lambda_ * np.eye(true_sample_size)) , axis = 1)

        return approximated_ls

    def recursive_ridge_leverage(self,
                                 X,
                                 n_components,
    ):
        """
        Recursive ridge leverage score sampling algorithm for the Nyström method.

        This function computes an approximation of the ridge leverage score, using a recursive strategy.
        Reference: Cameron Musco, Christopher Musco. Recursive Sampling for the Nyström Method. NIPS 2017.
        https://doi.org/10.48550/arXiv.1605.07583
        Based on the implementation of Axel Vanraes: https://github.com/axelv/recursive-nystrom

        Args:
            X: Numpy array of size (n, d) where n is the number of data and d number of features.
            n_components: Size of sub-matrix.
        """

        n_oversample = np.log(n_components)
        k = np.ceil(n_components / (4 * n_oversample)).astype(np.integer)
        n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(np.integer)
        perm = np.random.permutation(X.shape[0])

        # set up sizes for recursive levels
        size_list = [X.shape[0]]
        for l in range(1, n_levels+1):
            size_list += [np.ceil(size_list[l - 1] / 2).astype(np.integer)]

        # indices of poitns selected at previous level of recursion
        # at the base level it's just a uniform sample of ~ n_component points
        sample = np.arange(size_list[-1])
        indices = perm[sample]
        weights = np.ones((indices.shape[0],))

        # we need the diagonal of the whole kernel matrix
        k_diag = np.ones((X.shape[0],1))

        # Main recursion, unrolled for efficiency
        for l in reversed(range(n_levels)):
            # indices of current uniform sample
            current_indices = perm[:size_list[l]]
            # build sampled kernel

            # all rows and sampled columns
            KS = pairwise_kernels(X[current_indices,:], X[indices,:],
                                   metric=self.kernel,
                                   filter_params=True,
                                   n_jobs=self.n_jobs,
                                   **self._get_kernel_params())
            SKS = KS[sample, :] # sampled rows and sampled columns

            # optimal lambda for taking O(k log(k)) samples
            if k >= SKS.shape[0]:
                # for the rare chance we take less than k samples in a round
                lmbda = 10e-6
                # don't set to exactly 0 to avoid stability issues
            else:
                # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
                # can be interpret as the zoom level
                lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
                        - np.sum(eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k
            lmbda = np.maximum(lmbda, 1e-7)

            # compute and sample by lambda ridge leverage scores
            R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
            R = np.matmul(KS, R)
            #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
            if l != 0:
                # max(0, . ) helps avoid numerical issues, unnecessary in theory
                leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                        k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
                # on intermediate levels, we independently sample each column
                # by its leverage score. the sample size is n_components in expectation
                sample = np.where(np.random.uniform(size=size_list[l]) < leverage_score)[0]
                # with very low probability, we could accidentally sample no
                # columns. In this case, just take a fixed size uniform sample
                if sample.size == 0:
                    leverage_score[:] = n_components / size_list[l]
                    sample = np.random.choice(size_list[l], size=n_components, replace=False)
                weights = np.sqrt(1. / leverage_score[sample])

            else:
                leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                        k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
                p = leverage_score/leverage_score.sum()

                sample = np.random.choice(X.shape[0], size=n_components, replace=False, p=p)
            indices = perm[sample]

        return indices
