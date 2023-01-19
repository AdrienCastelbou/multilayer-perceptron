from Layer import *

class MLP():
    def __init__(self, hidden_layer_sizes=4, max_iter=300):
        self.n_layers = hidden_layer_sizes + 2
        self.max_iter = max_iter
        self.network = None

    def create_network_(self, n_inputs, n_outputs):
        self.network = []
        self.network.append(Dense(n_inputs, 100))
        for i in range(self.n_layers - 1):
            self.network.append(Dense(100, 100))
            self.network.append(ReLU())
        self.network.append(Dense(100, n_outputs))

    def forward(self, X):
        activations = []
        input = X
        for l in self.network:
            activations.append(l.forward(input))
            input = activations[-1]
        assert len(activations) == len(self.network)
        return activations


    def predict(self, X):
        logits = self.forward(X)
        return logits.argmax(axis=-1)
    
    def fit(self, X, y):
        self.create_network_(X.shape[1], len(np.unique(y)))
        layer_activations = self.forward(X)
        layers_input = [X] + layer_activations
        logits = layer_activations[-1]