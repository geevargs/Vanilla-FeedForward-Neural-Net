import os
import sys
import numpy as np
from matplotlib import pyplot as plt
###Neural Network - FeedForward Neural Network Implementation###########
#Sujith Mathew Geevarghese
def sigmoid(x):
    # activation function
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        total_args = len(sys.argv)
        if total_args < 3:
        print('usage: Neural_Network [No.of Weights]\n'
              ' e.g.: Neural_Network 4 \n')
        sys.exit(False)
        #weights
        weights = sys.argv[1]
        layers = sys.argv[2]
        self.weights1   = np.random.rand(self.input.shape[1],weights) 
        self.weights2   = np.random.rand(weights,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        #forward propagation through our network
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
	self.weights2 += d_weights2
	if __name__ == "__main__":
   	       X = np.array([[0,0,1],
                         [0,1,1],
                         [1,0,1],
                         [1,1,1]])
           y = np.array([[0],[1],[1],[0]])
           neural_net = NeuralNetwork(X,y)
    
    	for i in range(1500): # trains the NN 1,500 times
            neural_net.feedforward()
            neural_net.backprop()
            print(neural_net.output)
            print "Input: \n" + str(X)
            print "Predicted Output: \n" + str(output)
            print "Actual Output: \n" + str(y)
            print "Loss: \n" + str(np.mean(np.square(y - neural_net.feedforward(X)))) # mean sum squared loss
 	        print "\n

        
