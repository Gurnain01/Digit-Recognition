import numpy as np

"""Return the sigmoid value or activation value of our neuron i:e it maps any value to a value between 0 to 1"""
def sigmoid(a):
    return 1/(1+np.exp(-a))

"""Return the derivative of the signoid function"""
def deriv_sigmoid(a):
    return a*(1-a)

def train(inputs,output,weight0,weight1):
    
    a = inputs.T
    b = output.T
    """Giving the input to our network and calculating the outptut and storing it in l2"""
   
    b1 = sigmoid(np.dot(a,weight0))
    b2 = sigmoid(np.dot(b1,weight1))
    
    """Calculating the error by subtracting our Networks output from expected output"""
    error = b - b2 
    
     """This gives how much did our output layer contributed in our missed output"""
    b2_del = error * deriv_sigmoid(b2)
    
    """Calculating the error of out hidden layer"""
    error0 = b2_del.dot(weight1.T)
    
    """This gives how much did our hidden layer contributed in our missed output"""
    b1_del = error0 * deriv_sigmoid(b1)
    
    """updating the values of our weights by how much we missed"""
    weight1 += np.dot(b1.T,b2_del)
    weight0 += np.dot(a.T,b1_del)

    return weight0,weight1
