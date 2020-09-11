import sys


def write_results_txt(name, heading, data):
    with open(name, 'w') as file:
        file.write(heading)
        file.write(data)


def parse_input(exp_input):
    is_parameter = False
    fun_name = ""
    interval = [0, 0]
    mesh_types = ""
    input_type = ""
    input_d = ""
    param_name = ""
    params = ""

    for row in exp_input:
        row = row.split(" ")

        if row[0] == "fun":
            fun_name = row[1]
        elif row[0] == "interval":
            interval[0] = int(row[1])
            interval[1] = int(row[2])
        elif row[0] == "mesh":
            mesh_types = row[1:]
        elif row[0] == "input_t":
            input_type = row[1]
        elif row[0] == "input":
            if input_type == "list":
                try:
                    input_d = list(map(int, row[1:]))
                except ValueError:
                    sys.exit("Incorrect input data: Should be a enumeration of integers")
            elif input_type == "range":
                left_border = int(row[1])
                right_border = int(row[2])
                input_d = range(left_border, right_border)
            else:
                sys.exit("Incorrect input type name")
        elif row[0] == "parameter":
            is_parameter = True
        elif is_parameter:
            param_name = row[0]
            try:
                params = list(map(float, row[1:]))
            except ValueError:
                sys.exit("Parameters should be numeric")
            is_parameter = False
        else:
            sys.exit("Incorrect key word in a row. List of available key words:\n" 
                     "fun\n" "mesh\n" "input_t\n" "input\n" "parameter\n")

    return fun_name, interval, mesh_types, input_d, param_name, params
