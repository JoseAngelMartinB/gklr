from sklearn.kernel_approximation import Nystroem

from gklr.kernel_utils import *

class KernelMatrix():
    def __init__(self, X, choice_column, obs_column, attributes, kernel_params, Z=None):
        # TODO: Check arguments
        # Create a new kernel based on the kernel type selected
        self._kernel_params = None
        self._kernel = None
        self._K = None
        self.nystrom = False
        self.nystrom_compression = DEFAULT_NYSTROM_COMPRESSION
        self.alternatives = None
        self.K_per_alternative = dict()
        self.alt_to_index = dict() # Links each alternative with an index
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
        if "compression" in kernel_params:
            self.nystrom_compression = kernel_params["compression"]
            del kernel_params["compression"]

        if self.nystrom == True:
            # TODO: Check that Z is None
            pass
        else:
            if "n_jobs" in kernel_params:
                # TODO: Implement n_jobs (parallelization)
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
                    nystrom_kernel = Nystroem(kernel=kernel_type, n_components=nystrom_components, **kernel_params)
                    K_aux = nystrom_kernel.fit_transform(X_alt)
                else:
                    K_aux = self._kernel(Z_alt, X_alt, **kernel_params).astype(DEFAULT_DTYPE)
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
                choice_indices.append(self.alt_to_index[choice])
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
                if alt in self.alt_to_index.keys():
                    return self._K[self.K_per_alternative[self.alt_to_index[alt]]]
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