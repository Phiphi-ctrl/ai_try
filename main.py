from Neural_Network_class import Neural_Network
import numpy as np


if __name__ == '__main__':

    X = np. array([[2, 6], [4, 1], [3, 1], [5, 6]])
    y = np.array([[1], [0], [1], [1]])

    jarvis = Neural_Network(X,y,learning_rate = 0.6, maximal_loss = 0.000001)

    #jarvis.tasks('train')
    jarvis.tasks('train')