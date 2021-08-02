#from sklearn.kernel_approximation import Nystroem
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.linalg import svd
from sklearn.cluster import KMeans, MiniBatchKMeans

from gklr.kernel_utils import *

class KernelMatrix():
    def __init__(self, X, choice_column, obs_column, attributes, kernel_params, Z=None):
        # TODO: Check arguments
        # Create a new kernel based on the kernel type selected
        self._kernel_params = None
        self._kernel = None
        self._K = None
        self.nystrom = False
        self.nystrom_sampling = "uniform"
        self.nystrom_compression = DEFAULT_NYSTROM_COMPRESSION
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
            if "nystrom_sampling" in kernel_params:
                self.nystrom_sampling = kernel_params["nystrom_sampling"]
                del kernel_params["nystrom_sampling"]
        if "compression" in kernel_params:
            self.nystrom_compression = kernel_params["compression"]
            del kernel_params["compression"]

        if self.nystrom == True:
            # TODO: Check that Z is None
            pass
        else:
            if "n_jobs" in kernel_params:
                del kernel_params["n_jobs"]
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
                    nystrom_components = int(X_alt.shape[0] * self.nystrom_compression)
                    nystrom_kernel = Nystroem(kernel=kernel_type, n_components=nystrom_components,
                                              sampling=self.nystrom_sampling, **kernel_params)
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


class Nystroem():
    def __init__(self, kernel="rbf", *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100,
                 sampling="uniform", random_state=None, n_jobs=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.sampling = sampling

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, **fit_params).transform(X)

    def fit(self, X, y=None):
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
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)

        if self.sampling == "uniform":
            inds = rnd.permutation(n_samples)
            basis_inds = inds[:n_components]
            basis = X[basis_inds]
        elif self.sampling == "kmeans":
            kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=rnd)
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
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        return params
