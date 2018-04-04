import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

def deriv_sigmoid(a):
    return a*(1-a)

def train(inputs,output,weight0,weight1):
    
    a = inputs.T
    b = output.T
    
    b1 = sigmoid(np.dot(a,weight0))
    b2 = sigmoid(np.dot(b1,weight1))

    error = b - b2 

    b2_del = error * deriv_sigmoid(b2)

    error0 = b2_del.dot(weight1.T)

    b1_del = error0 * deriv_sigmoid(b1)

    weight1 += np.dot(b1.T,b2_del)
    weight0 += np.dot(a.T,b1_del)

    return weight0,weight1
