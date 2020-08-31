import numpy as np
from scipy.interpolate import make_lsq_spline, CubicSpline


class VariationalApproach:
    def __init__(self, x_samples, y_samples, knots, test_x, test_y, shift):
        self.x = x_samples
        self.y = y_samples
        self.test_x = test_x
        self.test_y = test_y
        self.knots = knots
        self.shift = shift
        self.best_knots, self.lsq_spline, self.tol, self.cubic_spl_err = self.find_best_knots()

    degree = 3
    Zero = 1.0e-12
    err_bound = 0

    # algorithm
    def find_best_knots(self):
        # step 1
        cubic_spl = CubicSpline(self.x, self.y)

        err_on_cubic_spl = max(abs(cubic_spl(self.test_x) - self.test_y))

        self.err_bound = err_on_cubic_spl

        ranks = self.rank_knots(cubic_spl, self.knots)

        min_n = len(self.knots) // 10
        knots, lsq_spline, err_spl = self.find_n_best_knots(cubic_spl, ranks, min_n)

        return knots, lsq_spline, err_spl, err_on_cubic_spl

    # step 2
    def rank_knots(self, spline, knots):
        shift = self.shift
        ranks = np.zeros(len(knots))

        jumps = np.array(2 * abs(spline(knots[1:-1] + shift[1], 3) - spline(knots[1:-1] - shift[0], 3)))

        for index in range(1, len(knots) - 1):
            delta = (knots[index + 1] - knots[index - 1]) ** 3

            ranks[index] = jumps[index - 1] * delta

        return ranks

    # step 3-5
    def find_n_best_knots(self, init_spline, ranks, min_n):
        test_x = self.test_x
        knots = self.knots
        last_knot = len(knots) - 1
        indices_max = ranks.argsort()

        new_knots = []
        lsq_spline = ()
        err_spl = 0

        for num in range(3, max(last_knot, len(self.x))):
            indices = np.sort(indices_max[-num:][::-1])

            new_knots = [knots[i] for i in indices if i != last_knot and i != 0]
            # new_knots = [knots[i] for i in indices
            #             if i != last_knot and i != 0 and abs(init_spline(knots[i], 1)) >= self.Zero]
            new_knots = np.array(new_knots)

            new_knots = np.r_[(self.x[0],) * (self.degree + 1)
                              , new_knots
                              , (self.x[-1],) * (self.degree + 1)]

            lsq_spline = make_lsq_spline(self.x, self.y, new_knots)

            err_spl = max(abs(lsq_spline(test_x) - init_spline(test_x)))

            if err_spl <= self.err_bound:
                return new_knots[self.degree + 1: -(self.degree + 1)], lsq_spline, err_spl

        return new_knots[self.degree + 1: -(self.degree + 1)], lsq_spline, err_spl
