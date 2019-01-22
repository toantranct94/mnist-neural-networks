import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import gzip

from keras.datasets import mnist

# from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

class neuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # predict the neural network
    def predict(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def save_weights(self):
        np.savetxt('weights.txt', np.vstack([self.who,self.wih]), delimiter=" ", fmt="%s") 
        pass


if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # number of input, hidden and output nodes
    # input 28x28 pixel
    input_nodes = 784
    hidden_nodes = 200
    # outputs from 0...9
    output_nodes = 10

    # learning rate
    learning_rate = 0.1

    nn = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    # go through all records in the training data set

    for image, label in zip(X_train, y_train):
        # scale and shift the inputs
        image = np.array(image)
        image = image.flatten()
        inputs = (np.asfarray(image) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(label)] = 0.99
        nn.train(inputs, targets)
        pass

    # predict
    image = np.array(X_test[2])
    image = image.flatten()
    _input = (np.asfarray(image) / 255.0 * 0.99) + 0.01
    _output = nn.predict(_input)
    label = np.argmax(_output)

    print("Predict: {}".format(label))
    print("Actual: {}".format(y_test[2]))
    nn.save_weights()

    