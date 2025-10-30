# Neural Network: MNIST Digit Classification
# Feedforward neural network with backpropagation

import math
import random
from typing import List, Tuple

class NeuralNetwork:
    """
    Multi-layer feedforward neural network with backpropagation.
    Implements a simple feed-forward network for image classification.
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        """
        Initialize neural network with specified architecture.
        
        Args:
            layer_sizes: List of integers defining nodes in each layer
                        [input_size, hidden1_size, hidden2_size, output_size]
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases randomly
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # Xavier initialization for better convergence
            limit = math.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            weight_matrix = [[random.uniform(-limit, limit) 
                            for _ in range(layer_sizes[i])] 
                           for _ in range(layer_sizes[i+1])]
            self.weights.append(weight_matrix)
            
            bias_vector = [0.0 for _ in range(layer_sizes[i+1])]
            self.biases.append(bias_vector)
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + math.exp(-max(min(x, 500), -500)))
    
    def sigmoid_derivative(self, x: float) -> float:
        """Derivative of sigmoid function."""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x: float) -> float:
        """ReLU activation function."""
        return max(0, x)
    
    def relu_derivative(self, x: float) -> float:
        """Derivative of ReLU function."""
        return 1.0 if x > 0 else 0.0
    
    def softmax(self, x: List[float]) -> List[float]:
        """Softmax function for output layer."""
        exp_x = [math.exp(min(val, 500)) for val in x]
        sum_exp = sum(exp_x)
        return [val / sum_exp for val in exp_x]
    
    def forward_propagation(self, input_data: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Perform forward propagation through the network.
        
        Args:
            input_data: Input features
            
        Returns:
            Tuple of (activations, weighted_sums) for each layer
        """
        activations = [input_data]
        weighted_sums = []
        
        current_activation = input_data
        
        for layer in range(self.num_layers - 1):
            # Calculate weighted sum: z = W*a + b
            weighted_sum = []
            for j in range(self.layer_sizes[layer + 1]):
                z = sum(self.weights[layer][j][i] * current_activation[i] 
                       for i in range(len(current_activation)))
                z += self.biases[layer][j]
                weighted_sum.append(z)
            
            weighted_sums.append(weighted_sum)
            
            # Apply activation function
            if layer == self.num_layers - 2:  # Output layer
                current_activation = self.softmax(weighted_sum)
            else:  # Hidden layers
                current_activation = [self.relu(z) for z in weighted_sum]
            
            activations.append(current_activation)
        
        return activations, weighted_sums
    
    def backward_propagation(self, input_data: List[float], target: List[float]) -> Tuple[List, List]:
        """
        Perform backpropagation to compute gradients.
        
        Args:
            input_data: Input features
            target: Target output (one-hot encoded)
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        # Forward pass
        activations, weighted_sums = self.forward_propagation(input_data)
        
        # Initialize gradient lists
        weight_gradients = [[[] for _ in range(self.layer_sizes[i])] 
                           for i in range(1, self.num_layers)]
        bias_gradients = [[] for _ in range(self.num_layers - 1)]
        
        # Backward pass
        deltas = [None] * (self.num_layers - 1)
        
        # Output layer error
        output_layer = self.num_layers - 2
        output_error = [activations[-1][i] - target[i] 
                       for i in range(len(activations[-1]))]
        deltas[output_layer] = output_error
        
        # Hidden layer errors (backpropagate)
        for layer in range(self.num_layers - 3, -1, -1):
            error = [0.0] * self.layer_sizes[layer + 1]
            for i in range(self.layer_sizes[layer + 1]):
                for j in range(self.layer_sizes[layer + 2]):
                    error[i] += deltas[layer + 1][j] * self.weights[layer + 1][j][i]
                error[i] *= self.relu_derivative(weighted_sums[layer][i])
            deltas[layer] = error
        
        # Compute gradients
        for layer in range(self.num_layers - 1):
            for j in range(self.layer_sizes[layer + 1]):
                for i in range(self.layer_sizes[layer]):
                    gradient = deltas[layer][j] * activations[layer][i]
                    if not weight_gradients[layer][j]:
                        weight_gradients[layer][j] = [0.0] * self.layer_sizes[layer]
                    weight_gradients[layer][j][i] = gradient
            
            bias_gradients[layer] = deltas[layer][:]
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, weight_gradients: List, bias_gradients: List):
        """
        Update network weights using computed gradients.
        
        Args:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        for layer in range(self.num_layers - 1):
            for j in range(self.layer_sizes[layer + 1]):
                for i in range(self.layer_sizes[layer]):
                    self.weights[layer][j][i] -= self.learning_rate * weight_gradients[layer][j][i]
                self.biases[layer][j] -= self.learning_rate * bias_gradients[layer][j]
    
    def train(self, X_train: List[List[float]], y_train: List[List[float]], 
              epochs: int = 50, batch_size: int = 32, verbose: bool = True) -> List[float]:
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X_train: Training input data
            y_train: Training target data (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print training progress
            
        Returns:
            List of accuracies per epoch
        """
        accuracies = []
        n_samples = len(X_train)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            total_loss = 0.0
            correct = 0
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Accumulate gradients for batch
                batch_weight_gradients = None
                batch_bias_gradients = None
                
                for idx in batch_indices:
                    weight_grads, bias_grads = self.backward_propagation(
                        X_train[idx], y_train[idx]
                    )
                    
                    if batch_weight_gradients is None:
                        batch_weight_gradients = weight_grads
                        batch_bias_gradients = bias_grads
                    else:
                        # Accumulate gradients
                        for layer in range(len(weight_grads)):
                            for j in range(len(weight_grads[layer])):
                                for i in range(len(weight_grads[layer][j])):
                                    batch_weight_gradients[layer][j][i] += weight_grads[layer][j][i]
                            for j in range(len(bias_grads[layer])):
                                batch_bias_gradients[layer][j] += bias_grads[layer][j]
                
                # Average gradients and update weights
                batch_len = len(batch_indices)
                for layer in range(len(batch_weight_gradients)):
                    for j in range(len(batch_weight_gradients[layer])):
                        for i in range(len(batch_weight_gradients[layer][j])):
                            batch_weight_gradients[layer][j][i] /= batch_len
                    for j in range(len(batch_bias_gradients[layer])):
                        batch_bias_gradients[layer][j] /= batch_len
                
                self.update_weights(batch_weight_gradients, batch_bias_gradients)
            
            # Calculate epoch accuracy and loss
            for i in range(n_samples):
                activations, _ = self.forward_propagation(X_train[i])
                prediction = activations[-1]
                
                # Cross-entropy loss
                loss = -sum(y_train[i][j] * math.log(max(prediction[j], 1e-10)) 
                          for j in range(len(prediction)))
                total_loss += loss
                
                # Check accuracy
                pred_class = prediction.index(max(prediction))
                true_class = y_train[i].index(max(y_train[i]))
                if pred_class == true_class:
                    correct += 1
            
            accuracy = correct / n_samples * 100
            accuracies.append(accuracy)
            avg_loss = total_loss / n_samples
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')
        
        return accuracies
    
    def predict(self, input_data: List[float]) -> int:
        """
        Make a prediction for given input.
        
        Args:
            input_data: Input features
            
        Returns:
            Predicted class index
        """
        activations, _ = self.forward_propagation(input_data)
        output = activations[-1]
        return output.index(max(output))
    
    def evaluate(self, X_test: List[List[float]], y_test: List[List[float]]) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test input data
            y_test: Test target data (one-hot encoded)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        correct = 0
        n_samples = len(X_test)
        
        confusion_matrix = [[0] * 10 for _ in range(10)]
        
        for i in range(n_samples):
            prediction = self.predict(X_test[i])
            true_class = y_test[i].index(max(y_test[i]))
            
            confusion_matrix[true_class][prediction] += 1
            
            if prediction == true_class:
                correct += 1
        
        accuracy = correct / n_samples * 100
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': n_samples,
            'confusion_matrix': confusion_matrix
        }


def generate_synthetic_mnist_data(n_samples: int = 1000) -> Tuple[List, List, List, List]:
    """
    Generate synthetic MNIST-like data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        digit = random.randint(0, 9)
        
        # Generate 28x28 pixel image (784 features) with digit pattern
        image = []
        for pixel in range(784):
            # Simple digit patterns
            row = pixel // 28
            col = pixel % 28
            
            # Create basic digit shapes
            value = 0.0
            if digit == 0:
                if (8 < row < 20) and (abs(col - 14) > 4):
                    value = random.uniform(0.7, 1.0)
            elif digit == 1:
                if 10 < col < 18:
                    value = random.uniform(0.7, 1.0)
            elif digit == 8:
                if (8 < row < 20) and (abs(col - 14) > 4 or abs(row - 14) > 4):
                    value = random.uniform(0.7, 1.0)
            else:
                if random.random() < 0.3:
                    value = random.uniform(0.5, 1.0)
            
            # Add noise
            value += random.uniform(-0.1, 0.1)
            image.append(max(0, min(1, value)))
        
        # One-hot encode label
        label = [0.0] * 10
        label[digit] = 1.0
        
        X_data.append(image)
        y_data.append(label)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * n_samples)
    X_train = X_data[:split_idx]
    y_train = y_data[:split_idx]
    X_test = X_data[split_idx:]
    y_test = y_data[split_idx:]
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Neural Network: MNIST Digit Classification")
    print("="*60)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic MNIST dataset...")
    X_train, y_train, X_test, y_test = generate_synthetic_mnist_data(1000)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create neural network: 784 -> 128 -> 64 -> 10
    print("\nInitializing Neural Network...")
    print("Architecture: [784, 128, 64, 10]")
    nn = NeuralNetwork([784, 128, 64, 10], learning_rate=0.01)
    
    # Train the network
    print("\nTraining Neural Network...")
    accuracies = nn.train(X_train, y_train, epochs=50, batch_size=32, verbose=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Correct Predictions: {results['correct']}/{results['total']}")
    
    print("\n" + "="*60)
    print("Neural Network Training Complete!")
