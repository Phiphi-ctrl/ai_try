import numpy as np
import os
import matplotlib.pyplot as plt

class Neural_Network:

    def __init__(self, X, y, learning_rate, maximal_loss):
        np.random.seed(0)
        self.input_size = len(X[0])
        self.hidden_size = 100
        self.output_size = len(y[0])
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.X = X
        self.y = y
        self.path_w2 = os.path.dirname(__file__)+ '/W2.txt'
        self.path_w1 = os.path.dirname(__file__)+ '/W1.txt'
        self.learning_rate = learning_rate 
        self.maximal_loss = maximal_loss
        self.pyplot_xaxis = []
        self.pyplot_yaxis = []

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def ddx_sigmoid(x):
        return x * (1 - x)

    def relu(x):
        return x

    # Definiere die Verwärtspropagation def forward(self):
    def forward(self):
        z1 = np.dot(self.X, self.W1) #to understand X * W1 = z1, sigmoid(z1) = a1, a1 * W2 = z2, sigmoid(z2) = a2, a2 * W3 = z3, sigmoid(z3) = a3
        a1 = Neural_Network.sigmoid(z1)
        z2 = np.dot(a1, self.W2)
        self.y_pred = Neural_Network.sigmoid(z2)


        return self.y_pred, a1

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
                self.y_pred, a1 = Neural_Network.forward(self)
        
                #Berechne den VerLust
                self.l = Neural_Network.loss(self)

                #berechne den Gradienten der Verlustfunktion
        
                grad_y_pred = 2 * (self.y_pred - self.y)
                grad_z2 = grad_y_pred * Neural_Network.ddx_sigmoid(self.y_pred)#self.y_pred * (1 - self.y_pred)
                grad_W2 = np.dot(a1.T, grad_z2)
    
                grad_a1 = np.dot(grad_z2, self.W2.T)
                grad_z1 = grad_a1 * Neural_Network.ddx_sigmoid(a1)
                grad_W1 = np. dot(self.X.T, grad_z1)

                

                # Aktualisiere die Gewichte und den Bias mithilfe des Gradientenabstiegs
                self.W1 -= self.learning_rate * grad_W1

                self.W2 -= self.learning_rate * grad_W2

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
                    file.write(f'{str(self.W1[i][j])}')

        with open(self.path_w2, 'w') as file:
            for i in range (0,len(self.W2)):
                if i !=0:
                    file.write('\n')
                for j in range(0, len(self.W2[i])):
                    file.write(f'{str(self.W2[1][j])}')
        
        print (f'Genome saved @: {self.path_w1} and {self.path_w2} modelcode: axd1f')
        
        return self.W1, self.W2


    def call_trained_model (self):
        
        print ('calling model weights')
        #call trained model:
        with open (self.path_w1, 'r') as file: 
            W1 = np.empty((self.input_size, self.hidden_size))
            for i, line in enumerate (file):
                line = line.split(' ')
                line. pop (-1)
                for j, weight in enumerate(line):
                    self.W1[i][j] = float(weight)

        with open (self.path_w2, 'r') as file: 
            W2 = np.empty((self.input_size, self.hidden_size))
            for i, line in enumerate (file):
                line = line.split(' ')
                line. pop (-1)
                for j, weight in enumerate(line):
                    self.W2[i][j] = float(weight)

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

        else:

            print('check your jarvis.tasks() argument')

if __name__ == '__main__':


    X = np. array([[2, 0], [4, 1], [3, 1], [4, 6]])
    y = np.array([[1], [0], [1], [1]])

    jarvis = Neural_Network(X,y,learning_rate = 0.4, maximal_loss = 0.000001)

    #jarvis.tasks('train')
    jarvis.tasks('train')

    
