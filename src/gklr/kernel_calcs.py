
from gklr.kernel_utils import *
from gklr.calcs import Calcs

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