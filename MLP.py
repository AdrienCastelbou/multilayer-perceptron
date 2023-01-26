from Layer import *
from matplotlib import pyplot as plt
class MLP():
    def __init__(self, hidden_layer_sizes=2, max_iter=300):
        self.n_layers = hidden_layer_sizes
        self.max_iter = max_iter
        self.network = []


    def softmax(self, X):
        new_X = np.array(X)
        for i in range(new_X.shape[0]):
            exp_x = np.exp(new_X[i] - np.max(new_X[i]))
            new_X[i] = exp_x / np.sum(exp_x)
        return new_X

    def get_preds_proba(self, X, y):
        p = self.softmax(X)
        preds = []
        for i in range(len(X)):
            preds.append([p[i][y[i]]])
        return np.array(preds).reshape(len(preds), -1)

    def loss(self, X, y):
        preds = self.get_preds_proba(X, y)
        log_likelihood = -np.log(preds)
        loss = np.sum(log_likelihood) / len(X)
        return loss
    

    def grad_softmax_crossentropy_with_logits(self, X, y):
        ones_for_answers = np.zeros_like(X)
        for i in range(len(X)):
            ones_for_answers[i,y[i][0]] = 1
        softmax = self.softmax(X)   
        return (- ones_for_answers + softmax) / X.shape[0]



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
        return logits.argmax(axis=-1).reshape(-1, 1)
    

    def train(self, X, y):
        layer_activations = self.forward(X)
        
        layer_inputs = [X]+layer_activations
        logits = layer_activations[-1]
        loss = self.loss(logits,y)
        loss_grad = self.grad_softmax_crossentropy_with_logits(logits,y)
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) 
        return np.mean(loss)

    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        train_log = []
        for epoch in range(2000):
            loss = self.train(X, y)
            train_log.append(np.mean(self.predict(X)==y))
            print("Epoch",epoch)
            print("Train accuracy:",train_log[-1], "loss:", loss)
        plt.plot(train_log,label='train accuracy')
        plt.legend(loc='best')
        plt.grid()
        plt.show()