import random
import numpy as np

protocol = ['tcp', 'udp', 'icmp']
flags = ['SF', 'S0', 'S1', 'S2', 'S3', 'OTH', 'REJ', 'RSTO',
         'RSTOS0', 'SH', 'RSTR', 'SHR']

data_f = []

# Range for scaling of parameters
a = 0.1
b = 0.99


def read_file(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(line.strip().split(','))

    return data


def numericalise_data(data):
    for i in range(len(data)):
        # Drop the last column
        data[i] = data[i][:-1]

        # Transform nonnumerical features to numerical
        data[i][1] = str(protocol.index(data[i][1]) + 1)
        data[i][3] = str(flags.index(data[i][3]) + 1)

        data[i][2] = '0'

        if data[i][41] == 'normal':
            data[i][41] = '1'
        else:
            data[i][41] = '-1'

    data = [list(map(float, x)) for x in data]
    return data


def get_amounts_of_data(data):
    n1 = 0
    n2 = 0
    for x in data:
        if x[41] == 1:
            n1 += 1
        else:
            n2 += 1
    return n1, n2


def balance_data(data, n1, n2, R):
    item_to_delete = -1 if R == n1 else 1
    data_new = []

    items_to_keep = R

    for i in range(len(data)):
        if data[i][41] == item_to_delete:
            items_to_keep -= 1
            if items_to_keep >= 0:
                data_new.append(data[i])
        else:
            data_new.append(data[i])

    return data_new


def separate_data(data):
    data = np.array(data)
    return data[:, 0:41], data[:, 41]


def norma(f_vector):
    x_max = max(f_vector)
    x_min = min(f_vector)

    for i in range(len(f_vector)):
        y = (f_vector[i] - x_min) / (x_max - x_min)
        y = (b - a) * y + a
        f_vector[i] = y

    return f_vector


def scale_features(data):
    for i in range(len(data)):
        data[i] = norma(data[i])

    return data


def pre_process_data(data_path, N):
    # Read file to memory
    data = read_file(data_path)

    # Shuffle data
    random.shuffle(data)

    # Slice data
    data_sample = data[:N]

    # Numericalise data
    data_sample = numericalise_data(data_sample)

    # Calculate amount of data: N1 as normal and N2 as attack
    N1, N2 = get_amounts_of_data(data_sample)

    # Find whether normal samples or attack samples are fewer
    R = min(N1, N2)

    # Balance data
    data_balanced = balance_data(data_sample, N1, N2, R)

    # Shuffle new training dataset
    random.shuffle(data_balanced)

    # Separate data from labels
    data_balanced, y_true = separate_data(data_balanced)

    # Feature scaling
    data_normalised = scale_features(data_balanced)

    return data_normalised, y_true


def numericalise_data_x(data):

    # Transform nonnumerical features to numerical
    data[1] = str(protocol.index(data[1]) + 1)
    data[3] = str(flags.index(data[3]) + 1)

    data[2] = '0'

    data = list(map(float, data))
    return data
