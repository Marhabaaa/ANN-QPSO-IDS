import sys
from modules import data_arrangement as data_arr
from modules import qpso
from modules import ann

# Define network topology
input_nodes = 41
hidden_nodes = 100

# Define sample size for trainig (N) and test (M)
N = 5000
M = 3000

# Define QPSO topology
number_of_particles = 20
max_iter = 30
# Set range of Creativity coefficient
a = 0.2
b = 0.95

# Set Pseudo-inverse penalty parameter
C = 99999999999

# Get data
train_data, y_train_data = data_arr.pre_process_data(
    'data/KDDTrain_20Percent.txt', N)
test_data, y_test_data = data_arr.pre_process_data('data/KDDTest.txt', M)

# Initialise ANN
annx = ann.ANN(input_nodes, hidden_nodes, train_data, y_train_data, C)

# Initialise QPSO
swarm = qpso.Swarm(number_of_particles, input_nodes*hidden_nodes, max_iter, a, b, annx.test_mse)
swarm.init_swarm()

# Get hidden weights (optimal weights given by QPSO)
weights = swarm.gbest.reshape(input_nodes, hidden_nodes)
annx.test(weights, test_data, y_test_data)

# Calculate output weights (Beta)
B = annx.get_beta(annx.get_H(train_data, weights), y_train_data)


x = '-1'
while x != 'exit':
    if x != 'exit':
        x = str(input('Ingrese datos a predecir: (ingrese exit para salir)'))
        x = x.strip().split(',')

        prediction = annx.predict(data_arr.norma(data_arr.numericalise_data_x(x)), weights, B)
        print()
        print(prediction)
        if prediction > 0:
            print('Normal')
        else:
            print('Ataque')

        print()
