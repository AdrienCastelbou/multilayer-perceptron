from Layer import *
from matplotlib import pyplot as plt
class MLP():
    def __init__(self, hidden_layer_sizes=2, max_iter=300):
        self.n_layers = hidden_layer_sizes
        self.max_iter = max_iter
        self.network = []


    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


    def softmax_crossentropy_with_logits(self, logits,reference_answers):
        print(logits)
        logits_for_answers = logits[np.arange(len(logits)),reference_answers]
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
        return xentropy
    

    def grad_softmax_crossentropy_with_logits(self, logits,reference_answers):
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)),reference_answers] = 1
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        return (- ones_for_answers + softmax) / logits.shape[0]



    def create_network_(self, n_inputs, n_outputs):
        self.network = []
        self.network.append(Dense(n_inputs, 4))
        self.network.append(ReLU())
        for i in range(self.n_layers - 1):
            self.network.append(Dense(4, 4))
            self.network.append(ReLU())
        self.network.append(Dense(4, n_outputs))
        self.n_layers += 2

    def forward(self, X):
        activations = []
        input = X
        for l in self.network:
            activations.append(l.forward(input))
            input = activations[-1]
        return activations


    def predict(self, X):
        if self.network == []:
            self.create_network_(X.shape[1], 2)
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)
    

    def train(self, X, y):
        layer_activations = self.forward(X)
        
        layer_inputs = [X]+layer_activations
        logits = layer_activations[-1]
        loss = self.softmax_crossentropy_with_logits(logits,y)
        loss_grad = self.grad_softmax_crossentropy_with_logits(logits,y)
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) 
        return np.mean(loss)

    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        train_log = []
        for epoch in range(200):
            loss = self.train(X, y)
            train_log.append(np.mean(self.predict(X)==y))
            print("Epoch",epoch)
            print("Train accuracy:",train_log[-1], "loss:", loss)
        plt.plot(train_log,label='train accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()