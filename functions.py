import numpy as np


class Functions:

    eps = 1

    def runge(self, x):
        return 1 / (1 + x ** 2)

    def cos_p_exp(self, x):
        return np.cos(np.pi * x / 2) + np.exp(-x / self.eps)

    def x_p_w(self, x):
        def w(var):
            return np.exp(-var / self.eps)

        return x + (w(x) - w(1)) / (1 - w(1))

    def parse_func_name(self, fun_name):
        if fun_name == "cos_p_exp":
            return self.cos_p_exp
        elif fun_name == "x_p_w":
            return self.x_p_w
        elif fun_name == "runge":
            return self.runge
