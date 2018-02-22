import numpy as np
import math


#
# constructor_a: create neural_network of size (n1,n2,...,nk) ,
#   n1 = number of neurons in the 1. layer, that is the number of input neurons
#
# constructor_(list of theta arrays):  # [THETA_ARRAY_1, THETA_ARRAY_2,...,THETA_ARRAY_3]
#   - The arrays of parameters essential define the design of the neural network
#   - Also would be used as the constructor to reload parameters
#       (which might / should have been saved to file)
#
#   BETTER: Since python does not distinguish by type
#           -> check what type the element has
#
#


class NeuralNetwork:
    thetas = []
    size = []

    def __init__(self, *args):
        if type(args[0]) is np.ndarray:
            # copy thetas from args
            for i in range(0, len(args), 1):
                theta_is_valid = True
                if i > 0:
                    # valid if num rows in last matrix equals num columns in new matrix
                    # for i=0, matrix num of columns should be bigger than one
                    # (at least one input neuron + one bias unit)
                    theta_is_valid = len(self.thetas[i - 1]) + 1 == len(args[i][0])
                else:
                    theta_is_valid = len(args[i][0]) > 1

                if theta_is_valid:
                    self.thetas.append(args[i])
                else:
                    raise Exception("Bad format of matrices!")
                    # check for validity of the passed matrices

                self.size.append(len(self.thetas[i][0]) - 1)
            self.size.append(len(self.thetas[len(args) - 1]))  # since index i accounts for neur. in layer (i - 1)

        if type(args[0]) is int:
            # create arrays of Theta in accordance with the number of neurons per layer.
            for i in range(0, len(args) - 1, 1):
                self.thetas.append(np.ones((args[i + 1], args[i] + 1)))
                self.size.append(args[i])
            self.size.append(args[len(args) - 1])  # since last index in loop only goes to len - 2

    def feed_forward(self, x):

        activation = np.array(x)

        # raising several exceptions
        if activation.ndim != 1:
            raise Exception("Bad format of input vector, only 1 dim supported")

        if len(activation) != len(self.thetas[0][0]) - 1:
            raise Exception("Bad format of input vector, len not according to theta matrix size")

        # perform feed forward. a: activation of current layer

        for theta in self.thetas:
            activation = self.sigmoid(np.dot(theta, np.append([1], activation)))

        return activation




    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(x))
    @staticmethod
    def sigmoid_d1(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))


a = np.ones((1, 2))
b = np.ones((1, 2))

# print(len(c[0]))

nn = NeuralNetwork(a,b)

activation = nn.feed_forward(np.array([1]))

print(activation)
print(nn.sigmoid(1 + nn.sigmoid(2)))