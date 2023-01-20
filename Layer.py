import numpy as np

class Layer():
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_inputs = np.eye(num_units)
        return grad_output @ d_layer_d_inputs



class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        relu_forward = []
        for i in input:
            relu_forward.append(max(0, i))
        relu_forward = np.array(relu_forward)
        return relu_forward
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad
    

class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1, status="hidden"):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(input_units + output_units)), size=(input_units, output_units))
        self.biases = np.zeros(output_units)
        self.status=status
    

    def forward(self, input):
        return input @ self.weights + self.biases
    

    def backward(self, input, grad_output):
        grad_input = grad_output @ self.weights.T
        grad_weights = input.T @ grad_output
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input