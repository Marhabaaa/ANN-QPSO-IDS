import numpy as np
import random


class Swarm:
    def __init__(self, number_of_particles, dimension, max_iteration, a, b, function):
        self.P = number_of_particles
        self.D = dimension
        self.pbest = None
        self.gbest = None
        self.error_pbest = np.full(number_of_particles, np.inf)
        self.error_gbest = np.inf
        self.mBest = None
        # Current iteration
        self.iteration = 0
        # Maximum number of iterations
        self.max_iter = max_iteration

        # alpha represents the Creativity coefficient with a as its minimum and b its maximum
        self.alpha = None
        self.a = a
        self.b = b

        # Function to test
        self.test_function = function

    def init_swarm(self):
        # Fill all the particles with random numbers between 0 and 1
        X = np.random.random((self.P, self.D))
        # Initialise pbest
        self.pbest = X.copy()

        # Iterate until max_iter times
        # Every iteration is a swarm update
        while self.iteration < self.max_iter:
            print()
            print('Iteration: ' + str(self.iteration + 1))
            print()

            # Calculate mBest
            self.calculate_mbest()
            # Calculate Creativity coefficient
            self.calculate_alpha()

            # Iterate for each particle
            for i in range(self.P):
                # Evaluate current solution in order to get the error
                current_error = self.test_function(X[i, :])

                # -----------------------------
                prev_error_gbest = self.error_gbest
                # -----------------------------

                # Update pBest
                if current_error < self.error_pbest[i]:
                    self.pbest[i, :] = np.copy(X[i, :])
                    self.error_pbest[i] = current_error

                # Update gBest
                if current_error < self.error_gbest:
                    self.gbest = np.copy(X[i, :])
                    self.error_gbest = current_error

                # -----------------------------
                print(str(i + 1) + '\t' + str('%.17f' % current_error) + '\t' + str('%.17f' % self.error_pbest[i])
                      + '\t' + str('%.17f' % self.error_gbest), end='')

                if self.error_pbest[i] == self.error_gbest:
                    print(' *', end='')

                    if self.error_gbest < prev_error_gbest:
                        print(' new')
                    else:
                        print()
                else:
                  print()
                # -----------------------------

                # Iterate for each dimension
                for j in range(self.D):
                    # Set random value for φ between 0 and 1
                    phi = random.random()
                    # Calculate the local atractor
                    p = self.local_attractor(phi, i, j)
                    # Set random value for μ between 0 and 1
                    mu = random.random()

                    # Update value j of particle i based on quantum behaviour
                    if random.random() > 0.5:
                        X[i, j] = p + self.alpha * abs(self.mBest[j] - X[i, j]) * np.log(1 / mu)
                    else:
                        X[i, j] = p - self.alpha * abs(self.mBest[j] - X[i, j]) * np.log(1 / mu)

            self.iteration += 1

            # -----------------------------
            print()
            print('Min MSE so far: ' + str(self.error_gbest))
            print('Best solution so far: [' + str('%.8f'%(self.gbest[0])) + ' ' + str('%.8f'%(self.gbest[1])) + ' ... ' + str('%.8f'%(self.gbest[2])) + ']')
            print()
            print('---------------------------------')
            # -----------------------------

    def calculate_alpha(self):
        self.alpha = (self.b - self.a) * ((self.max_iter - self.iteration) / self.max_iter) + self.a

    def calculate_mbest(self):
        self.mBest = []
        for j in range(self.D):
            self.mBest.append(np.sum(self.pbest[:, j]) / self.P)

    def local_attractor(self, phi, i, j):
        return phi * self.pbest[i, j] + (1 - phi) * self.gbest[j]
