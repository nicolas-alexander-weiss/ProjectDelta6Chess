import numpy as np


#
# constructor_a: create neural_network of size (n1,n2,...,nk) ,
#   n1 = number of neurons in the 1. layer, that is the number of input neurons
#
# constructor_(list of theta arrays):  # [THETA_ARRAY_1, THETA_ARRAY_2,...,THETA_ARRAY_3]
#   - The arrays of parameters essential define the design of the neural network
#   - Also would be used as the constructor to reload parameters
#       (which might / should have been saved to file)
#
#

class NeuralNetwork:
    def __init__(self):
