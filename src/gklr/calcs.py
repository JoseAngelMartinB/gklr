import numpy as np

class Calcs():
    def __init__(self, K):
        self.K = K

    def calc_probabilities(self):
        pass

    def log_likelihood(self):
        pass

    def calc_f(self):
        pass

    def calc_Y(self, f):
        return np.exp(f)

    def calc_G(self):
        pass

    def calc_P(self, Y, G, G_j):
        return (Y*G_j)/G