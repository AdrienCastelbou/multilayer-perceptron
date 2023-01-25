from Layer import *

class MLP():
    def __init__(self, hidden_layer_sizes=2, max_iter=300):
        self.n_layers = hidden_layer_sizes
        self.max_iter = max_iter
        self.network = []


    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


    def loss(self, X, y):
        m = X.shape[0]
        p = self.softmax(X) + 1e-10
        log_likelihood = -np.log(p[[range(m)],y])
        loss = np.sum(log_likelihood) / m
        return loss

    def grad_loss(self, X, y):
        m = y.shape[0]
        grad = self.softmax(X)
        grad[[range(m)],y] -= 1
        grad = grad/m
        print(grad)



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
        if self.network == []:
            self.create_network_(X.shape[1], 2)
        layer_activations = self.forward(X)
        layers_input = [X] + layer_activations
        logits = layer_activations[-1]
        loss = self.loss(logits, y)
        loss_grad = self.grad_loss(logits, y)    
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_grad = layer.backward(layers_input[layer_index], loss_grad)
        return np.mean(loss)

    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        layer_activations = self.forward(X)
        layers_input = [X] + layer_activations
        logits = layer_activations[-1]