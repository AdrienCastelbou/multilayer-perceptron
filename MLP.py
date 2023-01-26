from Layer import *
from matplotlib import pyplot as plt
from utils import data_spliter

class MLP():
    def __init__(self, hidden_layer_sizes=np.full(2, 100), max_iter=300, learning_rate=0.001):
        self.n_layers = hidden_layer_sizes
        self.max_iter = max_iter
        self.network = []
        self.loss_curve = []
        self.val_loss_curve = []
        self.learning_rate = learning_rate


    def softmax(self, X):
        new_X = np.array(X)
        for i in range(new_X.shape[0]):
            exp_x = np.exp(new_X[i] - np.max(new_X[i]))
            new_X[i] = exp_x / np.sum(exp_x)
        return new_X

    
    def loss(self, y_hat, y):
        v_ones = np.ones((y.shape[0], 1))
        return - float(y.T @ np.log(y_hat + 1e-15) + (v_ones - y).T @ np.log(1 - y_hat + 1e-15)) / y.shape[0]

    def grad_softmax_crossentropy(self, X, y):
        ones_for_answers = np.zeros_like(X)
        for i in range(len(X)):
            ones_for_answers[i,y[i][0]] = 1 
        return (- ones_for_answers + X) / X.shape[0]


    def create_network_(self, n_inputs, n_final_outputs):
        self.network = []
        for i in range(len(self.n_layers)):
            self.network.append(Dense(n_inputs, self.n_layers[i], learning_rate=self.learning_rate))
            self.network.append(ReLU())
            n_inputs = self.n_layers[i]
        self.network.append(Dense(n_inputs, n_final_outputs))

    def forward(self, X):
        activations = []
        input = X
        for l in self.network:
            activations.append(l.forward(input))
            input = activations[-1]
        activations[-1] = self.softmax(input)
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
        loss_grad = self.grad_softmax_crossentropy(logits,y)
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) 

    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        X_train, X_test, y_train, y_test = data_spliter(X, y, 0.8)
        for epoch in range(1, self.max_iter + 1):
            self.train(X_train, y_train)
            train_preds = self.predict(X_train)
            test_preds = self.predict(X_test)
            self.loss_curve.append(self.loss(train_preds, y_train))
            self.val_loss_curve.append(self.loss(test_preds, y_test))
            print(f"Epoch {epoch}/{self.max_iter} - loss: {self.loss(train_preds, y_train)} - val_loss: {self.loss(test_preds, y_test)} - accuracy: {np.mean(train_preds == y_train)} - val_accuracy: {np.mean(test_preds == y_test)} ")
