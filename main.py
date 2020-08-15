import numpy as np
from scipy.interpolate import LSQUnivariateSpline, CubicSpline
from prettytable import PrettyTable
import pandas as pd


class VariationalApproach:
    def __init__(self, x_samples, y_samples, knots, test_x, test_y, err):
        self.x = x_samples
        self.y = y_samples
        self.test_x = test_x
        self.test_y = test_y
        self.knots = knots
        self.err_bound = err
        self.best_knots, self.lsq_spline, self.tol = self.find_best_knots()

    shift = 0.1

    # algorithm
    def find_best_knots(self):
        # step 1
        cubic_spl = CubicSpline(self.x, self.y)
        # print(self.size_sample, len(self.knots), max(abs(cubic_spl(self.test_x) - self.test_y)))

        ranks = self.rank_knots(cubic_spl, self.knots)

        min_n = len(self.knots) // 10
        knots, lsq_spline, err_spl = self.find_n_best_knots(cubic_spl, ranks, min_n)

        return knots, lsq_spline, err_spl

    # step 2
    def rank_knots(self, spline, knots):
        ranks = np.zeros(len(self.knots))
        jumps = 2 * abs(spline(knots[1:-1] + self.shift, 3) - spline(knots[1:-1] - self.shift, 3))

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

        for num in range(min_n, last_knot):
            indices = np.sort(indices_max[-num:][::-1])

            new_knots = [knots[i] for i in indices if i != last_knot and i != 0]
            new_knots = np.array(new_knots)

            lsq_spline = LSQUnivariateSpline(self.x, self.y, new_knots)
            err_spl = max(abs(lsq_spline(test_x) - init_spline(test_x)))
            if err_spl < self.err_bound:
                return new_knots, lsq_spline, err_spl

        return new_knots, lsq_spline, err_spl


def runge_function(x):
    return 1 / (1 + x ** 2)


def sin_p_x_function(x):
    return x + np.sin(3*x)


def test_on_function(func, left_bound, right_bound, n_samples, init_n_knots, err_bound):
    x_samples = np.linspace(left_bound, right_bound, n_samples)
    test_x = np.linspace(left_bound, right_bound, n_samples * 3)

    knots = np.linspace(left_bound, right_bound, init_n_knots)

    y_samples = func(x_samples)
    test_y = func(test_x)

    alg = VariationalApproach(x_samples, y_samples, knots, test_x, test_y, err_bound)

    return len(alg.best_knots), alg.tol


def write_results_txt(name, heading, data):
    with open(name, 'w') as file:
        file.write(heading)
        file.write(data)


def write_results_xls(name, data, heading):
    df = pd.DataFrame(data, columns=heading)
    df.to_excel(excel_writer=name)


if __name__ == "__main__":
    headers = ["Num samples", "Initial num of knots", "Best num interior knots", "Tolerance"]
    table_runge = PrettyTable(headers)
    table_sin = PrettyTable(headers)
    table_sin_p_x = PrettyTable(headers)
    # table_runge = []
    # table_sin = []
    # table_sin_p_x = []

    for num_samples in range(20, 250, 10):
        for init_num_knots in range(num_samples // 3, num_samples - 3):
            best_knots, tolerance = test_on_function(runge_function, -5, 5, num_samples, init_num_knots, 1.0e-3)
            table_runge.add_row([num_samples, init_num_knots, best_knots, tolerance])
            # table_runge.append([num_samples, init_num_knots, best_knots, tolerance])

            best_knots, tolerance = test_on_function(np.sin, 0, 2*np.pi, num_samples, init_num_knots, 1.0e-3)
            table_sin.add_row([num_samples, init_num_knots, best_knots, tolerance])
            # table_sin.append([num_samples, init_num_knots, best_knots, tolerance])

            best_knots, tolerance = test_on_function(sin_p_x_function, -2, 2, num_samples, init_num_knots, 1.0e-3)
            table_sin_p_x.add_row([num_samples, init_num_knots, best_knots, tolerance])
            # table_sin_p_x.append([num_samples, init_num_knots, best_knots, tolerance])

    # write_results_xls("D:/variational_approach/results_runge_e-03.xlsx", table_runge, headers)
    # write_results_xls("D:/variational_approach/results_sin_e-03.xlsx", table_sin, headers)
    # write_results_xls("D:/variational_approach/results_sinpx_e-03.xlsx", table_sin_p_x, headers)

    write_results_txt('results_runge_e-03.txt', "runge results\n", table_runge.get_string())
    write_results_txt('results_sin_e-03.txt', "sinus results\n", table_sin.get_string())
    write_results_txt('results_sin_p_x_e-03.txt', "sinus plus x results\n", table_sin_p_x.get_string())
