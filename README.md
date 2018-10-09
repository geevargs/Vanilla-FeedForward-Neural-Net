# Vanilla-FeedForward-Neural-Net
Task is to create a vanilla neural network implementation with a variable number of layers and nodes per layer.

A simple feedforward neural network consists of :

1) Input layer x

2) Arbitary amount of hidden layers

3) An Output layer 

4) A Set of weights and biases(w &b)

5) Activation Function (Sigmoid Activation function)

This is a neural network that deos not use any of the deep learning library like TensorFlow and Caffe and is a simple two layer neural network architecture.

The accuracy and strength of the prediction is based on the the weights and biases which is the training process.
1) Calculating the predicted output ŷ, known as feedforward.


2) Updating the weights and biases, known as backpropagation.

In the first task I have declared biases as 0 for simplicity.

Training an Iris Dataset using a simple feedforward network which is also designed from scratch.

1)Split dataset into training and  testing dataset. The testing dataset is generally smaller than training one as it will help in training the model better.
The data contains four features — sepal length, sepal width, petal length, and petal width for the different species (versicolor, virginica and setosa) of the flower.
It is stored in a 150x4 numpy.ndarray.


2)Show the scatter plot of the iris dataset petal length vs petal width.


3)Initialize the Input,Hidden layer,Output which are the parameters and declare the weights and bias.


4)Define size of the parameters.


5)Implement Forward Proagation in which we initialise the sigmoid and tanh function.

6)Implement Backward Progation.


5) Compute the cost function









