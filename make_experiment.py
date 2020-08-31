from variational_approach import VariationalApproach
import numpy as np
from prettytable import PrettyTable
import pandas as pd
from functions import Functions


def make_uniform_mesh(left_b, right_b, step):
    return np.linspace(left_b, right_b, step)


def make_partially_uniform_mesh(init_n_knots, left_b, right_b, eps):
    knots = np.zeros(init_n_knots)
    num_intervals = init_n_knots - 1
    knots[0], knots[-1] = left_b, right_b

    alpha = 1
    sigma = min(0.5, 2*eps * np.log(num_intervals) / alpha)

    step = 2 * sigma / init_n_knots
    for i in range(0, init_n_knots - 2):
        if i >= num_intervals // 2:
            step = 2 * (1 - sigma) / init_n_knots

        knots[i + 1] = knots[i] + step

    return knots


def test_on_function(func, left_bound, right_bound, n_samples, init_n_knots, eps):
    x_samples = make_uniform_mesh(left_bound, right_bound, n_samples)
    # x_samples = make_partially_uniform_mesh(n_samples, left_bound, right_bound, eps)

    test_x = np.linspace(left_bound, right_bound, 1000)

    y_samples = func(x_samples)
    test_y = func(test_x)

    knots = make_uniform_mesh(left_bound, right_bound, init_n_knots)
    # knots = make_partially_uniform_mesh(init_n_knots, left_bound, right_bound, eps)

    left_shift = np.array([(knots[i] - knots[i - 1]) / 2 for i in range(1, init_n_knots - 1)])
    right_shift = np.array([(knots[i + 1] - knots[i]) / 2 for i in range(1, init_n_knots - 1)])

    alg = VariationalApproach(x_samples, y_samples, knots, test_x, test_y, (left_shift, right_shift))

    return len(alg.best_knots), alg.tol, alg.cubic_spl_err


def write_results_txt(name, heading, data):
    with open(name, 'w') as file:
        file.write(heading)
        file.write(data)


def make_experience():
    headers = ["Epsilon", "Num samples", "Initial num of knots", "Best num interior knots"
               , "Cubic tolerance", "Error between cubic and lsq spline"]
    file_names = ['test/uni/cos_exp_same_mesh_28_101.txt', 'test/uni/x_p_w_same_mesh.txt'
                  , 'test/uni/results_runge_28_100.txt']
    # paths = ["D:/variational_approach/" + file_names[i][:-4] + ".xlsx" for i in range(len(file_names))]
    names = ["cos_exp_same_mesh\n", "x_p_w_same_mesh\n"
             , "runge results\n"]
    intervals = [(0, 1), (0, 1), (-5, 5), (0, 2 * np.pi), (-2, 2), (-3, 3)]
    fun = Functions()
    functions = [fun.cos_p_exp, fun.x_p_w, fun.runge, np.sin, fun.sin_p_x, fun.sin_p_x2]

    table = PrettyTable(headers)

    for i in range(2, len(functions[:3])):
        for num_samples in [2 ** i + 1 for i in range(4, 9)]:
            init_num_knots = num_samples
            for eps in [1.0e-2]:
                fun.eps = eps
                num_best_knots, tolerance, cubic_spl_err = test_on_function(functions[i], intervals[i][0]
                                                                            , intervals[i][1], num_samples
                                                                            , init_num_knots, eps)

                table.add_row([eps, num_samples, init_num_knots, num_best_knots, cubic_spl_err, tolerance])

        write_results_txt(file_names[i], names[i], table.get_string())

        table.clear_rows()


if __name__ == "__main__":
    make_experience()
