import numpy as np


class Functions:

    eps = 1

    def runge(self, x):
        return 1 / (1 + x ** 2)

    def sin_p_x(self, x):
        return x + np.sin(3 * x)

    def sin_p_x2(self, x):
        return x ** 2 + np.sin(3 * x)

    def cos_p_exp(self, x):
        return np.cos(np.pi * x / 2) + np.exp(-x / self.eps)

    def x_p_w(self, x):
        def w(var):
            return np.exp(-var / self.eps)

        return x + (w(x) - w(1)) / (1 - w(1))
