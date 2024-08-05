from Neural_Network_class import Neural_Network
import numpy as np



if __name__ == '__main__':

    list_size = 30

    X = np.array([[0],[1],[2],[3],[4]])
    y = np.array([[1],[2],[3],[4],[5]])

    #

    lol = Neural_Network(X,y,learning_rate = 0.0001, maximal_loss = 0.00001, momentum=0.5)
    
    #jarvis.tasks('train')
    lol.tasks('train')