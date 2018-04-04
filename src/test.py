import network as net
import numpy as np


def feedforward(a,weight0,weight1):

    l = a.T
    b1 = net.sigmoid(np.dot(l,weight0))
    b2 = net.sigmoid(np.dot(b1,weight1))
    return b2;

def check(te_inputs,te_output,weight0,weight1):

    correct = 0
    
    for i in range(len(te_inputs)):
        
        out = feedforward(te_inputs[i],weight0,weight1)
        f_out = np.argmax(out)
        if(f_out == te_output[i]):
            correct += 1

    print ("Accuracy Of the Network is " , ((correct/10000)*100))
