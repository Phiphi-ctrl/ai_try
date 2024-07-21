from Neural_Network_class import Neural_Network
import numpy as np



if __name__ == '__main__':

    list_size = 10

    X = np.empty((list_size,1))
    y = np.empty((list_size,1))

    for i in range(0, list_size):
        X[i] = i
        y[i] = i * 30

    lol = Neural_Network(X,y,learning_rate = 0.0001, maximal_loss = 0.0001)
    
    #jarvis.tasks('train')
    lol.tasks('train')