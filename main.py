import numpy as np
from scipy.interpolate import make_lsq_spline, CubicSpline
from prettytable import PrettyTable
import pandas as pd


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

        self.err_bound = 2 * err_on_cubic_spl

        ranks = self.rank_knots(cubic_spl, self.knots)

        min_n = len(self.knots) // 100
        knots, lsq_spline, err_spl = self.find_n_best_knots(cubic_spl, ranks, min_n)

        return knots, lsq_spline, err_spl, err_on_cubic_spl

    # step 2
    def rank_knots(self, spline, knots):
        ranks = np.zeros(len(knots))
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
            # new_knots = [knots[i] for i in indices
            #             if i != last_knot and i != 0 and abs(init_spline(knots[i], 1)) >= self.Zero]
            new_knots = np.array(new_knots)

            new_knots = np.r_[(self.x[0],) * (self.degree + 1)
                              , new_knots
                              , (self.x[-1],) * (self.degree + 1)]

            lsq_spline = make_lsq_spline(self.x, self.y, new_knots)

            err_spl = max(abs(lsq_spline(test_x) - init_spline(test_x)))
            if err_spl < self.err_bound:
                return new_knots[self.degree + 1: -(self.degree + 1)], lsq_spline, err_spl

        return new_knots[self.degree + 1: -self.degree], lsq_spline, err_spl


def runge_function(x):
    return 1 / (1 + x ** 2)


def sin_p_x_function(x):
    return x + np.sin(3 * x)


def sin_p_x2_function(x):
    return x ** 2 + np.sin(3 * x)


def test_on_function(func, left_bound, right_bound, n_samples, init_n_knots, shift):
    x_samples = np.linspace(left_bound, right_bound, n_samples)
    test_x = np.linspace(left_bound, right_bound, n_samples * 10)

    knots = np.linspace(left_bound, right_bound, init_n_knots)

    y_samples = func(x_samples)
    test_y = func(test_x)

    alg = VariationalApproach(x_samples, y_samples, knots, test_x, test_y, shift)

    return len(alg.best_knots), alg.tol, alg.cubic_spl_err


def write_results_txt(name, heading, data):
    with open(name, 'w') as file:
        file.write(heading)
        file.write(data)


def write_results_xls(name, data, heading):
    df = pd.DataFrame(data, columns=heading)
    df.to_excel(excel_writer=name)


def make_experience():
    headers = ["Num samples", "Initial num of knots", "Best num interior knots"
               , "Shift", "Cubic tolerance", "Tolerance"]
    file_names = ['results_runge_100_1000.txt', 'results_sin_e-04.txt'
                  , 'results_sin_p_x_e-04.txt', 'results_sin_p_x2_e-04.txt']
    # paths = ["D:/variational_approach/" + file_names[:-4] + ".xlsx" for i in range(len(file_names))]
    names = ["runge results\n", "sinus results\n"
             , "sinus plus x results\n", "sinus plus square(x) results\n"]
    intervals = [(-5, 5), (0, 2 * np.pi), (-2, 2), (-3, 3)]
    functions = [runge_function, np.sin, sin_p_x_function, sin_p_x2_function]

    table = PrettyTable(headers)
    # table = []

    for i in range(0, len(functions[:1])):  # testing only runge function
        for num_samples in range(101, 102):
            for init_num_knots in range(1001, 1002):
                shift = (intervals[i][1] - intervals[i][0]) / (init_num_knots - 1) / 2
                num_best_knots, tolerance, cubic_spl_err = test_on_function(functions[i], intervals[i][0]
                                                                            , intervals[i][1], num_samples
                                                                            , init_num_knots, shift)

                table.add_row([num_samples, init_num_knots, num_best_knots, shift, cubic_spl_err, tolerance])
                # table.append([num_samples, init_num_knots, num_best_knots, shift, cubic_spl_err, tolerance])

        write_results_txt(file_names[i], names[i], table.get_string())
        # write_results_xls(paths[i], table, headers)

        table.clear_rows()
        # table = []


if __name__ == "__main__":
    make_experience()
