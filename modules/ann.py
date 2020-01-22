import numpy as np
import math
from modules import performance_evaluate


# Get h_j
def activation_function(z):
    h = z * math.exp((-0.5) * math.pow(z, 2))
    return h


# Get z
def get_z(x_vector, w_vector):
    z = np.subtract(x_vector, w_vector)
    z = np.linalg.norm(z)
    return z


# Get y_n
def y_predicted(B, H):
    return np.dot(B, H)


# Calculate MSE
def mse(T_vector, y_vector):
    return np.square(T_vector - y_vector).mean()


class ANN:
    def __init__(self, input_nodes, hidden_nodes, train_data, y_train_data, C):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.train_data = train_data
        self.y_train_data = y_train_data
        self.C = C

    # Calculate MSE for QPSO
    def test_mse(self, weights_matrix):
        weights_matrix = weights_matrix.reshape(self.input_nodes, self.hidden_nodes)
        H = self.get_H(self.train_data, weights_matrix)
        H = np.array(H)
        B = self.get_beta(H, self.y_train_data)

        y_vector = y_predicted(B, H)

        return mse(self.y_train_data, y_vector)

    # Get the H matrix values of hidden nodes
    def get_H(self, input_data, w_matrix):
        H = []
        for x_vector in input_data:
            H.append(self.get_h_j(x_vector, w_matrix))
        return np.transpose(H)

    # Get the h_j vector values of hidden nodes
    def get_h_j(self, x_vector, w_matrix):
        h_j = []
        for i in range(self.hidden_nodes):
            h_j.append(activation_function(get_z(x_vector, w_matrix[:, i])))
        return h_j

    # Calculate Beta
    def get_beta(self, H, T_vector):
        return np.dot(np.linalg.pinv(np.dot(H, np.transpose(H)) + np.identity(self.hidden_nodes)/self.C),
                      np.dot(H, np.transpose(T_vector)))

    # Predict whether attack or not for a given features vector
    def predict(self, x_vector, w_matrix, B):
        return np.dot(B, self.get_h_j(x_vector, w_matrix))

    def test(self, weight_matrix, test_data, y_test_data):
        H = self.get_H(test_data, weight_matrix)
        H = np.array(H)
        B = self.get_beta(H, y_test_data)

        y_vector = y_predicted(B, H)

        for i in range(len(y_vector)):
            if y_vector[i] >= 0:
                y_vector[i] = 1
            else:
                y_vector[i] = -1

        tp, tn, fp, fn = performance_evaluate.get_values(y_test_data, y_vector)

        p, r, f_s, a = performance_evaluate.get_metrics(tp, tn, fp, fn)

        confusion_matrix = performance_evaluate.get_confusion_matrix(y_test_data, y_vector)
        print(confusion_matrix)

        print(performance_evaluate.get_report(y_test_data, y_vector))

        print('---------------------------------')
        print('MSE: ', mse(y_test_data, y_vector))
        print()
        print('Precision: ', str(p))
        print('Recall: ', str(r))
        print('F-score: ', str(f_s))
        print('Accuracy: ', str(a))
        print('---------------------------------')
