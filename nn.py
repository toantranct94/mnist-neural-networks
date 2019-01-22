import numpy as np
from scipy.special import expit # sigmoid 
import matplotlib.pyplot as plt

from keras.datasets import mnist


class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        
        # link weight matrices, wih and who
        self.wih = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # learning rate
        self.lr = learningrate
        
        # activation function: sigmoid function
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
        np.savetxt('who.txt', self.who, delimiter=" ", fmt="%s") 
        np.savetxt('wih.txt', self.wih, delimiter=" ", fmt="%s") 
        pass
    
    def set_weights(self, who, wih):
        self.who = who
        self.wih = wih
        pass

class Predict:
    def __init__(self, isTrainning, neuralNetwork):
        self.isTrainning = isTrainning
        self.neuralNetwork = neuralNetwork

    def get_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return X_train, y_train, X_test, y_test
    
    def predict(self, _input):
        # scale the input: 0.01 -> 1.00
        _input = np.array(_input)
        _input = _input.flatten()
        _input = (np.asfarray(_input) / 255.0 * 0.99) + 0.01

        # use exist weights 
        if not self.isTrainning:
            who = np.loadtxt('who.txt')
            wih = np.loadtxt('wih.txt')
            self.neuralNetwork.set_weights(who, wih)
            _output = self.neuralNetwork.predict(_input)
            label = np.argmax(_output)
            return label
        # train 
        else:
            print('Start training')
            X_train, y_train, X_test, y_test = self.get_data()
            for image, label in zip(X_train, y_train):
                # scale inputs: 0.01 -> 1.00
                image = np.array(image)
                image = image.flatten()
                inputs = (np.asfarray(image) / 255.0 * 0.99) + 0.01

                targets = np.zeros(output_nodes) + 0.01
                targets[int(label)] = 0.99
                self.neuralNetwork.train(inputs, targets)
            print('Finish training')
            # save weights:
            self.neuralNetwork.save_weights()
            # predict output:
            _output = self.neuralNetwork.predict(_input)
            label = np.argmax(_output)
            # accuracy:
            accuracy = self.accuracy(X_test, y_test)
            print('accuracy: ' + str(accuracy))
            return label

    def accuracy(self, X_test, y_test):
        acc_arr = []
        for image, label in zip(X_test, y_test):
            # scale inputs: 0.01 -> 1.00
            image = np.array(image)
            image = image.flatten()
            _input = (np.asfarray(image) / 255.0 * 0.99) + 0.01
            _output = self.neuralNetwork.predict(_input)
            _output = np.argmax(_output)
            if _output == label:
                acc_arr.append(1)
            else:
                acc_arr.append(0)
        
        acc = acc_arr.count(1) / len(X_test)
        return acc

    def show_predict(self, predict_label, actual_label,actual_image):
        plt.imshow(actual_image, cmap='gray')
        plt.title('Predict label: ' + str(predict_label) + ' --- ' + 'Actual label: ' + str(actual_label))
        plt.show()

if __name__ == "__main__":
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    isTrainning = False

    # number of input, hidden and output nodes
    # inputs: 28x28=784
    input_nodes = 784
    hidden_nodes = 200
    # outputs: 0...9
    output_nodes = 10
    # learning rate
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

    predict = Predict(isTrainning, nn)

    index_text = 2

    label = predict.predict(X_test[index_text])
    
    print("Predict: {}".format(label))
    print("Actual: {}".format(y_test[index_text]))

    predict.show_predict(label,y_test[index_text],X_test[index_text])


    
