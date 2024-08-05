from Neural_Network_class import Neural_Network
import numpy as np
import matplotlib.pyplot as plt

#function to graph the function created by the network
def graph_network(plot_range_max): 
    x_axis = []
    y_axis = []
    for i in range(0, plot_range_max):

        X = np.array([[i]])
        lol = Neural_Network(X,y,learning_rate = 0.01, maximal_loss = 0.000001,momentum=0.9)
        y_axis.append(lol.tasks('call')[0][0])
        x_axis.append(i)
            
    plt.plot(x_axis, y_axis)
    plt.show()

if __name__ == '__main__': 


    X = np.array([[10]])


    y = np.empty((1,1))

    lol = Neural_Network(X,y,learning_rate = 0.01, maximal_loss = 0.000001,momentum=0.9)
    
    #jarvis.tasks('train')
    lol.tasks('call')

    graph_network(19)

        
