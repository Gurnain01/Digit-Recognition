import numpy as np
import mnist_loader as ml
import network as net
import test as tt
import bar as br

np.random.seed(1)

"""randomly intitalising weights mapping input layer to hidden layer and
    hidden layer to output layer"""

weight0 = 2*np.random.random((784,50)) - 1
weight1 = 2*np.random.random((50,10)) - 1

"""Loading the MNIST data set into three lists two containg the training data
    and the third one containing the test data"""

tr_data, val_data, test_data = ml.load_data()

"""Fitting the 28*28 input image into a numpy array of 784*1 dimension"""

tr_inputs = [np.reshape(a, (784, 1)) for a in tr_data[0]]

"""Converting the single output into a numpy array of 10 dimensions with 1 at
    the index of the output an 0 elsewhere"""
tr_output = [ml.vectr_result(a) for a in tr_data[1]]

"""Loop to train the data taking an input of 10,000 images"""
for i in range(50000):

    weight0 , weight1 = net.train(tr_inputs[i],tr_output[i],weight0,weight1)
    if(i % 500) == 0 :
        br.progress(i, 50000)
br.progress(50000, 50000, cond = True)

print ("\n")

print ("Network Trained and ready to be operated")

te_inputs = [np.reshape(a, (784,1)) for a in test_data[0]]
te_output = test_data[1]

"""Function to check the accuracy of our trained network by testing it on
    unchecked data of 10,000 images"""
tt.check(te_inputs,te_output,weight0,weight1)

