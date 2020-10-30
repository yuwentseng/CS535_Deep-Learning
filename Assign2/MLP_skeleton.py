"""
CS535 Assignment#2
Name: YuWenTseng
ONID: 933652910
Mail: tsengyuw@oregonstate.edu

"""
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
import math
#from scipy.special import expit
#from sympy import *

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    def __init__(self, W, b):
    	# DEFINE __init function
        self.W = np.random.uniform(-1,1,(W, b))/10
        #self.W = np.random.randn(W, b) / 10
        self.b = np.random.uniform(-1,1,(1,b))/10
        #self.b = np.random.randn(1, b) / 10

    def forward(self, x):
    	# DEFINE forward function
        return np.dot(x,self.W) + self.b #batch_size*hidden
	
    def backward(self, x, learning_rate=0.0, momentum=0.0):
    	# DEFINE backward function
        return x                                

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self,x):
    	# DEFINE forward function
        relu_Forward = np.maximum(0,x)
        return relu_Forward 

    def backward(self,x,learning_rate=0.0, momentum=0.0):
    	# DEFINE backward function
        #When you take derivative, the entries that are negative have a derivative of 0, the entries that are positive have a derivative of 1, the entries that are 0 have a subderivative between [0,1].
        x[x > 0] = 1
        x[x < 0] = 0
        #x[x == 0] = 0
        x[x == 0] = np.random.uniform(0, 1)
        return x

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
# Reference from : https://deepnotes.io/softmax-crossentropy
class SigmoidCrossEntropy(object):  
    def forward(self, x, y):
        m = y.shape[0] #Or m=y.shape[1]?
        delta = 1e-3 #learning rate
        sigmoid = 1/(1+np.exp(-x))


        sigmoid[ sigmoid <= delta ] += delta 
        sigmoid[ sigmoid > delta] -= delta 

        cross_Entropy_Loss = -(y*np.log(sigmoid)+(1-y)*np.log(1-sigmoid))
        cross_Entropy_Cost = np.squeeze(np.sum(cross_Entropy_Loss) / m) #Reference from: https://towardsdatascience.com/deep-neural-networks-from-scratch-in-python-451f07999373

        return sigmoid, cross_Entropy_Cost

    def backward(self, sigmoidZ2, y,learning_rate=0.0,momentum=0.0):
        # DEFINE backward function
        return sigmoidZ2 - y  

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, train_input_dims, hidden_units, learning_rate, momentum, batch, D):

        #Initialize
        self.learning_rate = learning_rate
        self.batch = batch
        self.momentum = momentum
        self.D = D

        #Building the two layer and the function
        #On a fully connected layer, each neuron's output will be a linear transformer of the previous layer, composed with a non-linear activation function
        #Relu and Sigmoid
        self.layer1 =  LinearTransform(train_input_dims, hidden_units)
        self.function1 = ReLU()
        self.layer2 = LinearTransform(hidden_units, 1)
        self.function2 = SigmoidCrossEntropy()
        
        #Set the weight for two layers
        self.w = self.layer1.W
        self.w2 = self.layer2.W

        #np.tile => function
        #>>> c = np.array([1,2,3,4])
        #>>> np.tile(c,(4,1))
        #array([[1, 2, 3, 4],
        #[1, 2, 3, 4],
        #[1, 2, 3, 4],
        #[1, 2, 3, 4]])
        self.b = np.tile(self.layer1.b, (batch, 1))
        self.c = np.tile(self.layer2.b, (batch, 1))
        #Transfer the data type to float64 (astype)
        self.b = self.b.astype(np.float64)
        self.c = self.c.astype(np.float64)

        #Define z1,x2,z2, ouput
        self.z1 = 0
        self.x2 = 0
        self.z2 = 0
        self.output = 0

    # INSERT CODE for initializing the network

    def train(self, x_batch, y_batch):

        ####forward######

        #batch_size*hidden
        self.z1 = self.layer1.forward(x_batch)
        #batch_size*hidden
        #input layer -> hidden layer (Relu)
        self.x2 = self.function1.forward(self.z1) 
        #batch_size*1                 
        self.z2 = self.layer2.forward(self.x2)
        #hidden layer -> output layer (SigmoidCrossEntropy)
        output, entropy_loss = self.function2.forward(self.z2, y_batch)    

        #####backward########

        #Reference from the Assigment#1
        #dL_z2 = self.function2.backward(output,y_batch)    #batch_size*1
        #dL_z1 = np.multiply(np.dot(dL_z2,self.layer2.T),self.function1.backward(self.z1))    #feature_num*hidden 1*hidden.T
        #dz2_w2 = self.layer2.backward(self.x2)
        #dz2_z1 = self.function1.backward(self.z1)
        dz1_w = self.layer1.backward(x_batch)
        #dz2_w2.T, dL_z2
        dL_w2 = np.dot(self.layer2.backward(self.x2).T, self.function2.backward(output,y_batch)) 
        #dL_z2
        dL_c = self.function2.backward(output,y_batch)  
        #dL_z2, self.w2, dz2_z1
        dL_z1 = np.multiply((self.function2.backward(output,y_batch)) * (self.layer2.W.T), self.function1.backward(self.z1))
        #feature_num*hidden
        #dz1_w, dL_z1
        dL_w = np.dot(dz1_w.T, dL_z1)   
        #dL_z2, self.w2, dL_z1
        dL_b = np.multiply((self.function2.backward(output,y_batch)) * (self.layer2.W.T), self.function1.backward(self.z1)) 


        ######Updating weights and bias#############
        ###The constant term has a different gradient than the others (1)! If you are writing them separately, try avoid updating the constant as well.
        #####The Normal situation updates weights and bias: w = w + learning_rate * dw
        ####Momentum####
        ###w = w +(mu * d - learning_rate * g)
        if momentum: 
            #Update weight
            self.w += self.momentum * self.D[1] - self.learning_rate * dL_w
            #Update bias
            self.b += self.momentum * self.D[3] - self.learning_rate * dL_b
            #Update weight2
            self.w2 += self.momentum * self.D[0] - self.learning_rate * dL_w2
            #Update c
            #self.c += self.momentum * self.D[2] - self.learning_rate * dL_c
            
        else:
            self.w += self.learning_rate*dL_w
            self.b += self.learning_rate*dL_b
            self.w2 += self.learning_rate*dL_w2
            #self.c += self.learning_rate*dL_c
            
        return output, entropy_loss

    @staticmethod
    def evaluate(x, y):
        x[x >= 0.5] = 1
        x[x < 0.5] = 0
        return np.sum(x == y)

	# INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data']
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']

    # normalization
    train_x = (train_x - train_x.mean()) / train_x.std()
    test_x = (test_x - test_x.mean()) / test_x.std()

    train_num_examples, train_input_dims = train_x.shape    #10000, dim:3072
    test_num_examples, test_input_dims = test_x.shape       #10000, dim:3072

    
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 50
    num_batches = 10
    hidden_units = 50
    learning_rate = 1e-3
    momentum = 0.5
    #batch_list = [10, 50, 100, 250, 500]
    #rate_list = [1e-6, 1e-5, 1e-4, 1e-3]
    #hidden_units_list = [10, 50, 100, 150, 200]
    #momentum_list = [.5, .6, .7, .8, .9]

    test_list_accuracy = []
    train_list_accuracy = []
    test_list_loss = []
    train_list_loss = []

    # for hidden_units in hidden_units_list:
    D = np.zeros(4)
    mlp = MLP(train_input_dims, hidden_units, learning_rate, momentum, num_batches, D)

    result={'train_loss':[],'train_accuracy':[],'test_loss':[],'test_accuracy':[]}

    for epoch in range(num_epochs):
        train_loss, train_correct, test_loss, test_correct=0 ,0 ,0 ,0
        max_accuracy = 0
        max_epoch = 0

        ##########train#############
        for b in range(0, train_num_examples, num_batches):
            trainx = train_x[b: b + num_batches]
            trainy = train_y[b: b + num_batches]

            output, entropy_loss = mlp.train(trainx.astype(np.float128), trainy)

            train_correct += mlp.evaluate(output, trainy)
            train_loss += entropy_loss
        ############test#############
        for b in range(0, test_num_examples, num_batches):
            testx= test_x[b: b + num_batches]
            testy= test_y[b: b + num_batches]

            output, test_loss = mlp.train(testx.astype(np.float128), testy)
            
            test_correct += mlp.evaluate(output, testy)
            test_loss += entropy_loss

            if 100 * test_correct  / test_num_examples>max_accuracy:
                max_accuracy=100 * test_correct  / test_num_examples
                max_epoch=epoch

        result['train_loss'].append(train_loss / train_num_examples)
        result['train_accuracy'].append(100 * train_correct / train_num_examples)
        result['test_loss'].append(test_loss / test_num_examples)
        result['test_accuracy'].append(max_accuracy)

        print(
            '\r[Epoch {}]  '.format(
                    epoch + 1,
                ),
                end='',
        )
        #UPDATE train total_loss and accuracy
        print(
            '\r[Train :] train.Loss = {:.3f}, train_accuracy = {:.2f}'.format( 
                train_loss / train_num_examples,
                100 * train_correct / train_num_examples
                ),
                end = '',
        )
        train_list_accuracy.append(train_correct / train_num_examples)
        train_list_loss.append(train_loss / train_num_examples)
        # sys.stdout.flush()
        #UPDATE test total_loss and accuracy
        print(
            '\r[Test  :] test.Loss = {:.3f}, test_accuracy = {:.2f}'.format( 
                test_loss / test_num_examples, 
                100 * test_correct / test_num_examples
                ),
                end = '',
        )
        test_list_accuracy.append(test_correct / test_num_examples)
        test_list_loss.append(test_loss / test_num_examples)
    sys.stdout.flush()

    print("\r Result", result, end = '')
    
    '''
    plt.figure()
    plt.title('Result of Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Test epoch')

    plt.plot(train_list_accuracy, color='blue', linestyle = '-', label = 'Train Accuracy')
    plt.plot(test_list_accuracy, color='red', linestyle = '-',label = 'Test Accuracy')
    plt.xlim(0, 40)
    plt.ylim(0, 100) 
    plt.show()
    
    #plt.figure()
    #plt.plot(hidden_units_list, test_list_accuracy, color = 'red')
    #plt.plot(hidden_units_list, train_list_accuracy, color = 'blue')
    #plt.savefig('Batch_Size.jpg')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.show()

    # plt.figure()
    # plt.title("Different number of hidden_units\n")
    # plt.xlabel("Number of hidden_units")
    # plt.ylabel("Accuracy(%)")
    '''