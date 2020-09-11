from variational_approach import VariationalApproach
import numpy as np
from prettytable import PrettyTable
from functions import Functions
from file_ops import *
import sys


def make_uniform_mesh(left_b, right_b, step):
    return np.linspace(left_b, right_b, step)


def make_partially_uniform_mesh(init_n_knots, left_b, right_b, eps):
    knots = np.zeros(init_n_knots)
    num_intervals = init_n_knots - 1
    knots[0], knots[-1] = left_b, right_b

    alpha = 1
    sigma = min(0.5, 2 * eps * np.log(num_intervals) / alpha)

    step = 2 * sigma / init_n_knots
    for i in range(0, init_n_knots - 2):
        if i >= num_intervals // 2:
            step = 2 * (1 - sigma) / init_n_knots

        knots[i + 1] = knots[i] + step

    return knots


def test_on_function(func, left_bound, right_bound, mesh_type, n_samples, init_n_knots, eps):

    if mesh_type == "uniform":
        x_samples = make_uniform_mesh(left_bound, right_bound, n_samples)
        knots = make_uniform_mesh(left_bound, right_bound, init_n_knots)
    elif mesh_type == "shishkin":
        x_samples = make_partially_uniform_mesh(n_samples, left_bound, right_bound, eps)
        knots = make_partially_uniform_mesh(init_n_knots, left_bound, right_bound, eps)
    else:
        sys.exit("Incorrect mesh_type")

    test_x = np.linspace(left_bound, right_bound, 1000)

    y_samples = func(x_samples)
    test_y = func(test_x)

    left_shift = np.array([(knots[i] - knots[i - 1]) / 2 for i in range(1, init_n_knots - 1)])
    right_shift = np.array([(knots[i + 1] - knots[i]) / 2 for i in range(1, init_n_knots - 1)])

    alg = VariationalApproach(x_samples, y_samples, knots, test_x, test_y, (left_shift, right_shift))

    return len(alg.best_knots), alg.lsq_err, alg.cubic_spl_err


def make_experience(fun_name, interval, mesh_type, input_d, param_name, param=None):
    headers = ["Num samples", "Initial num of knots", "Best num interior knots"
               , "Cubic tolerance", "Error between cubic and lsq spline"]
    fun = Functions()

    table = PrettyTable(headers)
    file_name = "results/" + mesh_type + " mesh/" + fun_name + "_" + str(param) + "_results.txt"

    meta_info = fun_name + " with " + param_name + "=" + str(param) + "\n"

    for num_samples in input_d:
        init_num_knots = num_samples
        if param is not None:
            if param_name == "Epsilon":
                fun.eps = param

        function = fun.parse_func_name(fun_name)
        num_best_knots, tolerance, cubic_spl_err = test_on_function(function, interval[0], interval[1]
                                                                    , mesh_type, num_samples, init_num_knots
                                                                    , fun.eps)

        table.add_row([num_samples, init_num_knots, num_best_knots, cubic_spl_err, tolerance])
    write_results_txt(file_name, meta_info, table.get_string())

    table.clear_rows()


if __name__ == "__main__":
    try:
        file = open("input_data.txt", "r")
    except IOError:
        sys.exit("Cannot find the input data file")

    experiment_input = ""
    input_text = file.readlines()
    for index, row in enumerate(input_text):
        experiment_input += row

        if row == "\n" or index == len(input_text) - 1:
            non_empty_input = list(filter(lambda x: x != "", experiment_input.split("\n")))
            fun_name, interval, mesh_types, input_d, param_name, params = parse_input(non_empty_input)

            for mesh_type in mesh_types:
                if params != "":
                    for param in params:
                        make_experience(fun_name, interval, mesh_type, input_d, param_name, param)
                else:
                    make_experience(fun_name, interval, mesh_type, input_d, param_name)

            experiment_input = ""

    file.close()
