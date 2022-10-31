from typing import Optional, Any, Dict, List, Union

import pandas as pd
#from sklearn.kernel_approximation import Nystroem
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import svd
from sklearn.cluster import KMeans, MiniBatchKMeans

from .logger import *
from .config import Config
from .kernel_utils import *

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
            X: Train dataset stored in a pandas DataFrame.
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            config: A Config object that contains the hyperparameters of the GKLR model.
            Z: Test dataset stored in a pandas DataFrame. Default: None
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

        # Create the kernel matrix K
        self._init_kernel_matrix(X, choice_column, attributes, Z)

    def _init_kernel_matrix(self,
                            X: pd.DataFrame,
                            choice_column: str,
                            attributes: Dict[int, List[str]],
                            Z: Optional[pd.DataFrame] = None,
    ) -> None:
        """Construct and store the kernel matrix K.

        Args:
            X: Train dataset stored in a pandas DataFrame.
            choice_column: Name of the column of DataFrame `X` that contains the ID of chosen alternative.
            attributes: A dict that contains the columns of DataFrame `X` that are considered for each alternative.
                This dict is indexed by the ID of the available alternatives in the dataset and the values are list
                containing the names of all the columns considered for that alternative. 
            Z: Test dataset stored in a pandas DataFrame. Default: None
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
                    nystrom_components = int(X_alt.shape[0] * self.nystrom_compression)
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

    def get_alternatives(self) -> np.ndarray:
        """Return the available alternatives.

        Returns:
            A numpy array with the available alternatives.
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
            A numpy array with the choices per observation.
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

    def dot(self, A: np.ndarray, index: int = 0) -> np.ndarray:
        """Implements the dot product of the kernel matrix and numpy array A.

        Implements the matrix multiplication K ∙ A, where K is the kernel matrix 
        and A is a numpy array given as argument.

        Args:
            A: Numpy array to be multiplied by the kernel matrix.
            index: Index of the kernel matrix to be used.

        Returns:
            The dot product of the kernel matrix and `A`.
        """
        K = self.get_K(index=index)
        assert isinstance(K, np.ndarray)
        if self.nystrom:
            B = K.dot(K.T.dot(A))
        else:
            B = K.dot(A)
        return B


class Nystroem():
    def __init__(self,
                 kernel = "rbf",
                 *,
                 gamma = None,
                 coef0 = None,
                 degree = None,
                 kernel_params = None,
                 n_components = 100,
                 sampling = "uniform",
                 random_state = None,
                 n_jobs = None,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.sampling = sampling

    def fit_transform(self,
                      X,
                      y = None,
                      **fit_params,
    ):
        return self.fit(X, **fit_params).transform(X)

    def fit(self,
            X,
            y = None,
    ):
        """Fit estimator to data.
        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        """
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
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
        else:
            raise ValueError(
                "ERROR. {nystrom_sampling} is not a valid sampling strategy for Nyström method.".format(
                    nystrom_sampling=self.sampling))

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
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
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
