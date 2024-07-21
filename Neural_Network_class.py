import numpy as np
import os
import matplotlib.pyplot as plt

class Neural_Network:

    def __init__(self, X, y, learning_rate, maximal_loss):
        #np.random.seed(0)
        self.input_size = len(X[0])
        self.hidden_lay1_size = 40
        self.hidden_lay2_size = 40
        self.output_size = len(y[0])
        self.W1 = np.random.rand(self.input_size, self.hidden_lay1_size)
        self.W2 = np.random.rand(self.hidden_lay1_size, self.hidden_lay2_size)
        self.W3 = np.random.rand(self.hidden_lay2_size, self.output_size)
        self.X = X
        self.y = y
        self.path_w2 = os.path.dirname(__file__)+ '/W2.txt'
        self.path_w1 = os.path.dirname(__file__)+ '/W1.txt'
        self.path_w3 = os.path.dirname(__file__)+ '/W3.txt'
        self.learning_rate = learning_rate 
        self.maximal_loss = maximal_loss
        self.pyplot_xaxis = []
        self.pyplot_yaxis = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def ddx_sigmoid(x):
        return x * (1 - x)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return np.where(z > 0, 1, 0)

    # Definiere die Verwärtspropagation def forward(self):
    def forward(self):

        self.z1 = np.dot(self.X, self.W1) #to understand X * W1 = z1, sigmoid(z1) = a1, a1 * W2 = z2, sigmoid(z2) = a2, a2 * W3 = z3, sigmoid(z3) = a3
        self.a1 = Neural_Network.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = Neural_Network.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3)
        self.y_pred = Neural_Network.relu(self.z3)

    # Definiere die Verlustfunktion (hier verwenden wir den mittleren quadratischen Fehler)
    def loss (self):
        return np.mean((self.y_pred - self.y) ** 2)


    def train(self):

        self.l = 1
        iteration = 0
        #Trainiere das ModeLL mithilfe des Gradientenabstiegs
        try:

            while(self.l > self.maximal_loss):
                #Vorwärtspropagation
                Neural_Network.forward(self)
        
                #Berechne den VerLust
                self.l = Neural_Network.loss(self)

                m = self.y.shape[0]
                #backpropagation
                grad_y_pred = 2 * (self.y_pred - self.y) / m
                grad_z3 = grad_y_pred * Neural_Network.relu_derivative(self.z3)
                grad_W3 = np.dot(self.a2.T, grad_z3)

                grad_a2 = np.dot(grad_z3, self.W3.T)
                grad_z2 = grad_a2 * Neural_Network.ddx_sigmoid(self.a2) #grad_z2 = 
                grad_W2 = np.dot(self.a1.T, grad_z2)
    
                grad_a1 = np.dot(grad_z2, self.W2.T)
                grad_z1 = grad_a1 * Neural_Network.ddx_sigmoid(self.a1)
                grad_W1 = np. dot(self.X.T, grad_z1)

                # Aktualisiere die Gewichte und den Bias mithilfe des Gradientenabstiegs
                self.W1 -= self.learning_rate * grad_W1
                self.W2 -= self.learning_rate * grad_W2
                self.W3 -= self.learning_rate * grad_W3

                #plot learning behaviour
                self.pyplot_xaxis.append(iteration)
                self.pyplot_yaxis.append(self.l)

                iteration += 1
                
                print(f'loss: {Neural_Network.loss(self)}')

                #print （F'Loss： ｛Neural_Network. Loss（self）まり）
        except KeyboardInterrupt:
            pass
        print(f'Model training: Successfull, Minimal Loss reached: {Neural_Network.loss(self)}')


    def save_genome(self):

        #die Ecainienten weights und biases speichern
        with open(self.path_w1, 'w') as file: 
            for i in range(0, len(self.W1)):
                if i != 0:
                    file.write('\n')
                for j in range(0, len(self.W1[i])):
                    file.write(f'{str(self.W1[i][j])}' + ' ')

        with open(self.path_w2, 'w') as file:
            for i in range (0,len(self.W2)):
                if i !=0:
                    file.write('\n')
                for j in range(0, len(self.W2[i])):
                    file.write(f'{str(self.W2[i][j])}' + ' ')

        with open(self.path_w3, 'w') as file:
            for i in range (0,len(self.W3)):
                if i !=0:
                    file.write('\n')
                for j in range(0, len(self.W3[i])):
                    file.write(f'{str(self.W3[i][j])}' + ' ')
        
        print (f'Genome saved @: {self.path_w1} and {self.path_w2} modelcode: axd1f')
        
        return self.W1, self.W2, self.W3


    def call_trained_model (self):
        
        print ('calling model weights')
        #call trained model:
        with open (self.path_w1, 'r') as file: 
            self.W1 = np.empty((self.input_size, self.hidden_lay1_size))
            for i, line in enumerate (file):
                line = line.split(' ')
                line. pop (-1)
                for j, weight in enumerate(line):
                    self.W1[i][j] = float(weight)

        with open (self.path_w2, 'r') as file: 
            self.W2 = np.empty((self.hidden_lay1_size, self.hidden_lay2_size))
            for i, line in enumerate (file):
                line = line.split(' ')
                line. pop (-1)
                for j, weight in enumerate(line):
                    self.W2[i][j] = float(weight)

        with open (self.path_w3, 'r') as file: 
            self.W3 = np.empty((self.hidden_lay2_size, self.output_size))
            for i, line in enumerate (file):
                line = line.split(' ')
                line. pop (-1)
                for j, weight in enumerate(line):
                    self.W3[i][j] = float(weight)

        print (f'Genome called @: {self.path_w1} and {self.path_w2} modelcode: axdif*)')

        return self.W1, self.W2

    def print_y(self):
        print (f'Ouput Neurons: ')
        print (self.y_pred)

    def check_prop(self):
        print('Propagation: Successfull')

    def graph_learn(self): 

        # Create the plot
        plt.plot(self.pyplot_xaxis, self.pyplot_yaxis)

        # Add titles and labels
        plt.title('Loss over time')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # Display the plot
        plt.show()
    

    def tasks(self, task):

        taskchain_train = [Neural_Network.train, 
                        Neural_Network.save_genome, 
                        Neural_Network.forward, 
                        Neural_Network.check_prop, 
                        Neural_Network.print_y, 
                        Neural_Network.graph_learn]

        taskchain_call_network = [Neural_Network.call_trained_model, 
                                Neural_Network.forward, 
                                Neural_Network.print_y]

        if(task == 'train'):
            for func in taskchain_train:
                func(self)

        elif(task == 'call'):
            for func in taskchain_call_network:
                func(self)

            return self.y_pred

        else:

            print('check your lol.tasks() argument')



    