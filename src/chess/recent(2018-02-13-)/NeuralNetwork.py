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

        # defining vectorized functions
        self.vect_sigmoid = np.vectorize(self.sigmoid)
        self.vect_sigmoid_d1 = np.vectorize(self.sigmoid_d1)

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
            a_plus_bias = np.append([1], activation)
            activation = self.vect_sigmoid(np.dot(theta, a_plus_bias))

        return activation


# X is element of R(m x n), m = num_training_examples, n = num_features
    # if m = 1, X needs to be 1 x n matrix and not just np.vector/array
    # NOT YET WORKING FOR MULTICLASS CLASSIFICATION!!!

    # OUTPUT: [COST, GRADIENTS]
    def cost_grad(self, X, y, lambd, only_cost = 0):
        # y_matrix = np.eye(self.size[len(self.size)-1])

        m, n = np.shape(X)
        num_layers = len(self.size)

        activations = []
        weighted_input = []
        activations.append(np.hstack((np.ones((m, 1)), X)))
        # feedforward
        for i in range(0,len(self.thetas),1):
            z = np.dot(activations[i], np.transpose(self.thetas[i]))
            weighted_input.append(z)

            a = np.hstack((np.ones((m, 1)), self.vect_sigmoid(z)))
            activations.append(a)

        activations[-1] = activations[-1][:,1:]

        cost = -(1/m) * np.sum(np.log(activations[-1])*y
                               + np.log(1 - activations[len(activations) - 1]*(1-y)))

        # add regularization
        if lambd != 0:
            theta_square_sum = 0
            for theta in self.thetas:
                theta_square_sum += sum(theta * theta)
            cost += (lambd / m / 2) * theta_square_sum

        if only_cost == 1:
            return cost

        # backpropagation

        rel_errors = []
        rel_errors.append(activations[-1] - y)
        for i in range(num_layers - 2, 0, -1):
            # print("rel_error[0],", rel_errors[0])
            # print("thetas[i]", self.thetas[i])
            # print("dotProduct", (i + 1), np.dot(rel_errors[0], self.thetas[i][:, 1:]) * self.vect_sigmoid_d1(weighted_input[i]))
            rel_error = np.dot(rel_errors[0], self.thetas[i][:, 1:]) * self.vect_sigmoid_d1(weighted_input[i])
            rel_errors.insert(0, rel_error)

        #print(rel_errors)

        gradients = []
        for i in range(0, len(self.thetas),1):
            grad = np.dot(np.transpose(rel_errors[i]), activations[i])
            if lambd != 0:
                grad += np.hstack((np.zeros((self.size[i+1], 1)), lambd * self.thetas[i][:, 2:]))
            grad /= m
            gradients.append(grad)

        return [cost, gradients]


    def numerical_gradient(self, X, y, lambd):
        num_grads = []

        epsilon = 1e-5
        for i in range(0,len(self.thetas),1):

            shape = np.shape(self.thetas[i])
            num_grads.append(np.zeros(shape))

            for j in range(0,shape[0],1):
                for k in range(0,shape[1],1):
                    self.thetas[i][j][k] -= epsilon
                    loss1 = self.cost(X, y, lambd)

                    self.thetas[i][j][k] += 2 * epsilon
                    loss2 = self.cost(X, y, lambd)

                    self.thetas[i][j][k] -= epsilon

                    num_grad = (loss2 - loss1) / (2 * epsilon)

                    num_grads[-1][j][k] = num_grad

        return num_grads


    def sigmoid(self, x):
        return 1 / (1 + math.exp(x))

    def sigmoid_d1(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


theta1 = np.ones((1, 2))
theta2 = np.ones((1, 2))

# print(len(c[0]))

nn = NeuralNetwork(theta1, theta2)

training_X = np.array([[1],[2],[4]])
training_Y = np.array([[0],[0],[1]])


cost, grads = nn.cost_grad(training_X, training_Y, lambd=0)

print("--------------------------------------------")

print("Num_neurons: ", nn.size)

print("Thetas: ")
for theta in nn.thetas:
    print(theta)

print("Thetas[0][:,2:]", nn.thetas[0][:,1:])

print("Cost: ", cost)

print("Gradients: ")
for grad in grads:
    print(grad)

num_grads = nn.numerical_gradient(training_X, training_Y, 0)

print("Num_Gradients")
for num_grad in num_grads:
    print(num_grad)



