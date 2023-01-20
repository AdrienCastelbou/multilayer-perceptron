from Layer import *

class MLP():
    def __init__(self, hidden_layer_sizes=2, max_iter=300):
        self.n_layers = hidden_layer_sizes
        self.max_iter = max_iter
        self.network = []


    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


    def loss(self, logits, y):
        logits_for_answer = logits[np.arange((len(logits))), y]
        xentropy = - logits_for_answer * np.log(np.sum(np.exp(logits), axis=-1))
        return xentropy

    def grad_loss(self, logits, y):
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)),y] = 1
        sftmax = self.softmax(logits)
        return (- ones_for_answers + sftmax) / logits.shape[0]



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
            activations.append(l.forward(input.reshape(-1,)))
            input = activations[-1]
        assert len(activations) == len(self.network)
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
        print(layer_activations)
        logits = layer_activations[-1] 
        loss = self.loss(logits, y)
        loss_grad = self.grad_loss(logits, y)    
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network(layer_index)
            loss_grad = layer.backward(layers_input[layer_index], loss_grad)
        return np.mean(loss)

    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        layer_activations = self.forward(X)
        layers_input = [X] + layer_activations
        logits = layer_activations[-1]