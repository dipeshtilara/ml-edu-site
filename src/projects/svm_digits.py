# Support Vector Machine: Handwritten Digit Recognition
# CBSE Class 12 AI - SVM Classification Project
# This project implements SVM from scratch for digit recognition

import math
import random
from typing import List, Tuple, Dict, Any, Optional

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("SUPPORT VECTOR MACHINE: HANDWRITTEN DIGIT RECOGNITION")
    print("CBSE Class 12 AI - SVM Classification Project")
    print("=" * 80)
    print()

class SVMKernel:
    """Kernel functions for SVM"""
    
    @staticmethod
    def linear(x1: List[float], x2: List[float]) -> float:
        """Linear kernel: K(x1, x2) = x1 • x2"""
        return sum(a * b for a, b in zip(x1, x2))
    
    @staticmethod
    def polynomial(x1: List[float], x2: List[float], degree: int = 3, coef: float = 1.0) -> float:
        """Polynomial kernel: K(x1, x2) = (coef + x1 • x2)^degree"""
        dot_product = sum(a * b for a, b in zip(x1, x2))
        return (coef + dot_product) ** degree
    
    @staticmethod
    def rbf(x1: List[float], x2: List[float], gamma: float = 1.0) -> float:
        """Radial Basis Function (RBF) kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)"""
        squared_distance = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return math.exp(-gamma * squared_distance)
    
    @staticmethod
    def sigmoid(x1: List[float], x2: List[float], alpha: float = 1.0, coef: float = 0.0) -> float:
        """Sigmoid kernel: K(x1, x2) = tanh(alpha * x1 • x2 + coef)"""
        dot_product = sum(a * b for a, b in zip(x1, x2))
        return math.tanh(alpha * dot_product + coef)

class SupportVectorMachine:
    """Complete SVM implementation using Sequential Minimal Optimization (SMO)"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3, coef=1.0, tolerance=1e-3, max_iter=1000):
        self.kernel_name = kernel
        self.C = C  # Regularization parameter
        self.gamma = gamma  # RBF kernel parameter
        self.degree = degree  # Polynomial kernel degree
        self.coef = coef  # Kernel coefficient
        self.tolerance = tolerance
        self.max_iter = max_iter
        
        # Model parameters
        self.alphas = None
        self.b = 0.0
        self.X_support = None
        self.y_support = None
        self.support_indices = []
        
        # Set kernel function
        self.kernel_func = self._get_kernel_function()
    
    def _get_kernel_function(self):
        """Get the appropriate kernel function"""
        if self.kernel_name == 'linear':
            return SVMKernel.linear
        elif self.kernel_name == 'polynomial':
            return lambda x1, x2: SVMKernel.polynomial(x1, x2, self.degree, self.coef)
        elif self.kernel_name == 'rbf':
            return lambda x1, x2: SVMKernel.rbf(x1, x2, self.gamma)
        elif self.kernel_name == 'sigmoid':
            return lambda x1, x2: SVMKernel.sigmoid(x1, x2, self.gamma, self.coef)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def _compute_kernel_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """Compute the kernel matrix for all training examples"""
        n_samples = len(X)
        kernel_matrix = [[0.0 for _ in range(n_samples)] for _ in range(n_samples)]
        
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i][j] = self.kernel_func(X[i], X[j])
        
        return kernel_matrix
    
    def _compute_error(self, i: int, X: List[List[float]], y: List[int], kernel_matrix: List[List[float]]) -> float:
        """Compute prediction error for example i"""
        prediction = self.b
        
        for j in range(len(X)):
            if self.alphas[j] > 0:
                prediction += self.alphas[j] * y[j] * kernel_matrix[i][j]
        
        return prediction - y[i]
    
    def _select_second_alpha(self, i1: int, errors: List[float]) -> int:
        """Select the second alpha for optimization (heuristic)"""
        max_error_diff = 0
        i2 = -1
        
        # Look for alpha with maximum error difference
        for i in range(len(errors)):
            if self.alphas[i] > 0 and self.alphas[i] < self.C:
                error_diff = abs(errors[i1] - errors[i])
                if error_diff > max_error_diff:
                    max_error_diff = error_diff
                    i2 = i
        
        # If no good second alpha found, select randomly
        if i2 == -1:
            i2 = random.randint(0, len(errors) - 1)
            while i2 == i1:
                i2 = random.randint(0, len(errors) - 1)
        
        return i2
    
    def _optimize_pair(self, i1: int, i2: int, X: List[List[float]], y: List[int], 
                      kernel_matrix: List[List[float]], errors: List[float]) -> bool:
        """Optimize a pair of alphas using SMO algorithm"""
        if i1 == i2:
            return False
        
        # Get current alphas and labels
        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1 = y[i1]
        y2 = y[i2]
        
        # Compute bounds
        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if L == H:
            return False
        
        # Compute eta (second derivative)
        eta = 2 * kernel_matrix[i1][i2] - kernel_matrix[i1][i1] - kernel_matrix[i2][i2]
        
        if eta >= 0:
            return False
        
        # Compute new alpha2
        alpha2_new = alpha2_old - (y2 * (errors[i1] - errors[i2])) / eta
        
        # Clip alpha2
        if alpha2_new >= H:
            alpha2_new = H
        elif alpha2_new <= L:
            alpha2_new = L
        
        # Check for significant change
        if abs(alpha2_new - alpha2_old) < 1e-5:
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        
        # Update bias
        b1 = (self.b - errors[i1] - y1 * (alpha1_new - alpha1_old) * kernel_matrix[i1][i1] -
              y2 * (alpha2_new - alpha2_old) * kernel_matrix[i1][i2])
        
        b2 = (self.b - errors[i2] - y1 * (alpha1_new - alpha1_old) * kernel_matrix[i1][i2] -
              y2 * (alpha2_new - alpha2_old) * kernel_matrix[i2][i2])
        
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new
        
        return True
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train SVM using Sequential Minimal Optimization"""
        print(f"Training SVM with {self.kernel_name} kernel...")
        
        n_samples = len(X)
        self.alphas = [0.0] * n_samples
        self.b = 0.0
        
        # Compute kernel matrix
        print("Computing kernel matrix...")
        kernel_matrix = self._compute_kernel_matrix(X)
        
        # SMO main loop
        print("Starting SMO optimization...")
        iteration = 0
        num_changed = 0
        examine_all = True
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            # Compute errors for all examples
            errors = [self._compute_error(i, X, y, kernel_matrix) for i in range(n_samples)]
            
            if examine_all:
                # Examine all examples
                for i in range(n_samples):
                    if self._examine_example(i, X, y, kernel_matrix, errors):
                        num_changed += 1
            else:
                # Examine non-bound examples
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        if self._examine_example(i, X, y, kernel_matrix, errors):
                            num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            if iteration % 100 == 0:
                print(f"SMO iteration {iteration}: {num_changed} pairs changed")
        
        # Store support vectors
        self.support_indices = [i for i in range(n_samples) if self.alphas[i] > 1e-6]
        self.X_support = [X[i] for i in self.support_indices]
        self.y_support = [y[i] for i in self.support_indices]
        
        print(f"Training completed after {iteration} iterations")
        print(f"Number of support vectors: {len(self.support_indices)}")
    
    def _examine_example(self, i1: int, X: List[List[float]], y: List[int], 
                        kernel_matrix: List[List[float]], errors: List[float]) -> bool:
        """Examine example i1 for optimization"""
        y1 = y[i1]
        alpha1 = self.alphas[i1]
        error1 = errors[i1]
        
        # Check KKT conditions
        r1 = error1 * y1
        
        if (r1 < -self.tolerance and alpha1 < self.C) or (r1 > self.tolerance and alpha1 > 0):
            # Select second alpha
            i2 = self._select_second_alpha(i1, errors)
            
            if self._optimize_pair(i1, i2, X, y, kernel_matrix, errors):
                return True
        
        return False
    
    def predict_sample(self, sample: List[float]) -> int:
        """Predict class for a single sample"""
        if self.X_support is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        decision_value = self.b
        
        for i, (support_vector, alpha, y_sv) in enumerate(zip(self.X_support, 
                                                              [self.alphas[j] for j in self.support_indices],
                                                              self.y_support)):
            if alpha > 0:
                kernel_value = self.kernel_func(sample, support_vector)
                decision_value += alpha * y_sv * kernel_value
        
        return 1 if decision_value >= 0 else -1
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Make predictions for multiple samples"""
        return [self.predict_sample(sample) for sample in X]
    
    def decision_function(self, X: List[List[float]]) -> List[float]:
        """Compute decision function values"""
        decision_values = []
        
        for sample in X:
            decision_value = self.b
            
            for i, (support_vector, alpha, y_sv) in enumerate(zip(self.X_support, 
                                                                  [self.alphas[j] for j in self.support_indices],
                                                                  self.y_support)):
                if alpha > 0:
                    kernel_value = self.kernel_func(sample, support_vector)
                    decision_value += alpha * y_sv * kernel_value
            
            decision_values.append(decision_value)
        
        return decision_values

class MultiClassSVM:
    """Multi-class SVM using One-vs-Rest strategy"""
    
    def __init__(self, **svm_params):
        self.svm_params = svm_params
        self.binary_classifiers = {}
        self.classes = []
    
    def fit(self, X: List[List[float]], y: List[int]):
        """Train multi-class SVM"""
        self.classes = sorted(list(set(y)))
        print(f"Training multi-class SVM for {len(self.classes)} classes...")
        
        for class_label in self.classes:
            print(f"\nTraining binary classifier for class {class_label}...")
            
            # Create binary labels (1 for current class, -1 for others)
            binary_y = [1 if label == class_label else -1 for label in y]
            
            # Train binary SVM
            classifier = SupportVectorMachine(**self.svm_params)
            classifier.fit(X, binary_y)
            
            self.binary_classifiers[class_label] = classifier
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Make multi-class predictions"""
        if not self.binary_classifiers:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        predictions = []
        
        for sample in X:
            # Get decision values from all binary classifiers
            decision_values = {}
            
            for class_label, classifier in self.binary_classifiers.items():
                decision_values[class_label] = classifier.decision_function([sample])[0]
            
            # Predict class with highest decision value
            predicted_class = max(decision_values.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        
        return predictions

class DigitDataGenerator:
    """Generate simplified handwritten digit dataset"""
    
    @staticmethod
    def generate_digit_pattern(digit: int, size: int = 8) -> List[List[int]]:
        """Generate a pattern for a specific digit"""
        patterns = {
            0: [[1,1,1,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,1,1,1]],
            
            1: [[0,0,1,0],
                [0,1,1,0],
                [0,0,1,0],
                [0,0,1,0],
                [0,0,1,0],
                [0,0,1,0],
                [1,1,1,1]],
            
            2: [[1,1,1,1],
                [0,0,0,1],
                [0,0,0,1],
                [1,1,1,1],
                [1,0,0,0],
                [1,0,0,0],
                [1,1,1,1]],
            
            3: [[1,1,1,1],
                [0,0,0,1],
                [0,0,0,1],
                [1,1,1,1],
                [0,0,0,1],
                [0,0,0,1],
                [1,1,1,1]],
            
            4: [[1,0,0,1],
                [1,0,0,1],
                [1,0,0,1],
                [1,1,1,1],
                [0,0,0,1],
                [0,0,0,1],
                [0,0,0,1]]
        }
        
        base_pattern = patterns.get(digit, patterns[0])
        
        # Resize pattern to desired size
        if size == len(base_pattern[0]):
            return base_pattern
        
        # Simple scaling (duplicate pixels)
        scale_factor = size // len(base_pattern[0])
        scaled_pattern = []
        
        for row in base_pattern:
            new_row = []
            for pixel in row:
                new_row.extend([pixel] * scale_factor)
            # Repeat row
            for _ in range(scale_factor):
                scaled_pattern.append(new_row[:])
        
        return scaled_pattern
    
    @staticmethod
    def add_noise(pattern: List[List[int]], noise_level: float = 0.1) -> List[List[int]]:
        """Add noise to digit pattern"""
        noisy_pattern = []
        
        for row in pattern:
            noisy_row = []
            for pixel in row:
                if random.random() < noise_level:
                    # Flip pixel
                    noisy_row.append(1 - pixel)
                else:
                    noisy_row.append(pixel)
            noisy_pattern.append(noisy_row)
        
        return noisy_pattern
    
    @staticmethod
    def pattern_to_features(pattern: List[List[int]]) -> List[float]:
        """Convert 2D pattern to feature vector"""
        features = []
        
        # Flatten pattern
        for row in pattern:
            features.extend(row)
        
        # Add some derived features
        height = len(pattern)
        width = len(pattern[0]) if pattern else 0
        
        # Density features
        total_pixels = height * width
        active_pixels = sum(sum(row) for row in pattern)
        density = active_pixels / total_pixels if total_pixels > 0 else 0
        features.append(density)
        
        # Symmetry features
        horizontal_symmetry = 0
        for i in range(height):
            for j in range(width // 2):
                if pattern[i][j] == pattern[i][width - 1 - j]:
                    horizontal_symmetry += 1
        
        vertical_symmetry = 0
        for i in range(height // 2):
            for j in range(width):
                if pattern[i][j] == pattern[height - 1 - i][j]:
                    vertical_symmetry += 1
        
        features.append(horizontal_symmetry / (height * width // 2))
        features.append(vertical_symmetry / (height * width // 2))
        
        # Center of mass
        total_mass = 0
        center_x = 0
        center_y = 0
        
        for i in range(height):
            for j in range(width):
                if pattern[i][j] == 1:
                    total_mass += 1
                    center_x += j
                    center_y += i
        
        if total_mass > 0:
            center_x /= total_mass
            center_y /= total_mass
        
        features.append(center_x / width)
        features.append(center_y / height)
        
        return features
    
    @staticmethod
    def generate_digit_dataset(n_samples_per_digit: int = 50, digits: List[int] = None) -> Tuple[List[List[float]], List[int]]:
        """Generate complete digit recognition dataset"""
        random.seed(42)  # For reproducible results
        
        if digits is None:
            digits = [0, 1, 2, 3, 4]  # Use first 5 digits for simplicity
        
        X = []
        y = []
        
        for digit in digits:
            print(f"Generating samples for digit {digit}...")
            
            base_pattern = DigitDataGenerator.generate_digit_pattern(digit, size=8)
            
            for _ in range(n_samples_per_digit):
                # Add noise to create variation
                noise_level = random.uniform(0.05, 0.2)
                noisy_pattern = DigitDataGenerator.add_noise(base_pattern, noise_level)
                
                # Convert to feature vector
                features = DigitDataGenerator.pattern_to_features(noisy_pattern)
                
                X.append(features)
                y.append(digit)
        
        # Shuffle the dataset
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
        
        return list(X), list(y)

class SVMEvaluator:
    """Evaluation metrics for SVM classification"""
    
    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy"""
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        return correct / len(y_true)
    
    @staticmethod
    def confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[int]) -> List[List[int]]:
        """Calculate confusion matrix"""
        n_classes = len(classes)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
        
        for true_label, pred_label in zip(y_true, y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            matrix[true_idx][pred_idx] += 1
        
        return matrix
    
    @staticmethod
    def classification_report(y_true: List[int], y_pred: List[int], classes: List[int]) -> Dict:
        """Generate classification report"""
        confusion_mat = SVMEvaluator.confusion_matrix(y_true, y_pred, classes)
        report = {}
        
        for i, class_label in enumerate(classes):
            tp = confusion_mat[i][i]
            fp = sum(confusion_mat[j][i] for j in range(len(classes))) - tp
            fn = sum(confusion_mat[i][j] for j in range(len(classes))) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': sum(1 for label in y_true if label == class_label)
            }
        
        return report

def split_data(X: List[List[float]], y: List[int], train_ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
    """Split data into training and testing sets"""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # Create indices and shuffle them
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def print_data_analysis(X: List[List[float]], y: List[int]):
    """Print comprehensive data analysis"""
    print("\n" + "=" * 60)
    print("DIGIT RECOGNITION DATASET ANALYSIS")
    print("=" * 60)
    
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    print(f"Number of digit samples: {n_samples}")
    print(f"Number of features per digit: {n_features}")
    
    # Class distribution
    from collections import Counter
    class_counts = Counter(y)
    print(f"\nDigit Distribution:")
    for digit, count in sorted(class_counts.items()):
        percentage = count / n_samples * 100
        print(f"Digit {digit}: {count:3d} samples ({percentage:5.1f}%)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"{'Feature Type':<20} {'Mean':<8} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    
    # Analyze different types of features
    feature_types = ['Pixel Values', 'Density', 'H-Symmetry', 'V-Symmetry', 'Center-X', 'Center-Y']
    
    # Pixel features (first 64 features for 8x8 image)
    pixel_features = [X[i][:64] for i in range(n_samples)]
    if pixel_features and pixel_features[0]:
        pixel_values = [val for sample in pixel_features for val in sample]
        print(f"{feature_types[0]:<20} {sum(pixel_values)/len(pixel_values):<8.3f} {min(pixel_values):<8.3f} {max(pixel_values):<8.3f}")
    
    # Other derived features
    for i, feature_type in enumerate(feature_types[1:], 64):
        if i < n_features:
            feature_values = [X[j][i] for j in range(n_samples)]
            mean_val = sum(feature_values) / len(feature_values)
            min_val = min(feature_values)
            max_val = max(feature_values)
            print(f"{feature_type:<20} {mean_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f}")

def print_evaluation_results(y_true: List[int], y_pred: List[int], classes: List[int], model: MultiClassSVM):
    """Print comprehensive evaluation results"""
    print("\n" + "=" * 60)
    print("SVM DIGIT RECOGNITION RESULTS")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = SVMEvaluator.accuracy(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Model information
    print(f"\nModel Configuration:")
    print(f"Kernel: {model.svm_params.get('kernel', 'rbf')}")
    print(f"C parameter: {model.svm_params.get('C', 1.0)}")
    print(f"Number of classes: {len(classes)}")
    
    # Support vector information
    total_support_vectors = sum(len(classifier.support_indices) for classifier in model.binary_classifiers.values())
    print(f"Total support vectors: {total_support_vectors}")
    
    # Confusion matrix
    confusion_mat = SVMEvaluator.confusion_matrix(y_true, y_pred, classes)
    print(f"\nConfusion Matrix:")
    header = "True\\Pred"
    print(f"{header:<10}", end="")
    for digit in classes:
        print(f"{digit:<6}", end="")
    print()
    
    for i, digit in enumerate(classes):
        print(f"{digit:<10}", end="")
        for j in range(len(classes)):
            print(f"{confusion_mat[i][j]:<6}", end="")
        print()
    
    # Classification report
    report = SVMEvaluator.classification_report(y_true, y_pred, classes)
    print(f"\nPer-Class Performance:")
    print(f"{'Digit':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for digit, metrics in report.items():
        print(f"{digit:<6} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['support']:<10d}")
    
    # Sample predictions
    print(f"\nSample Predictions (First 15 test cases):")
    print(f"{'True':<6} {'Pred':<6} {'Result':<8}")
    print("-" * 25)
    
    for i in range(min(15, len(y_true))):
        result = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"{y_true[i]:<6} {y_pred[i]:<6} {result:<8}")

def main():
    """Main function to demonstrate SVM for digit recognition"""
    print_header()
    
    # Generate digit recognition dataset
    print("Generating handwritten digit dataset...")
    X, y = DigitDataGenerator.generate_digit_dataset(n_samples_per_digit=60, digits=[0, 1, 2, 3, 4])
    
    # Analyze dataset
    print_data_analysis(X, y)
    
    # Split data
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Create and train SVM
    print("\n" + "=" * 60)
    print("TRAINING MULTI-CLASS SVM MODEL")
    print("=" * 60)
    
    # SVM with RBF kernel
    svm_model = MultiClassSVM(
        kernel='rbf',
        C=1.0,
        gamma=0.1,
        tolerance=1e-3,
        max_iter=500
    )
    
    svm_model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = svm_model.predict(X_test)
    
    # Evaluate model
    classes = sorted(list(set(y)))
    print_evaluation_results(y_test, y_pred, classes, svm_model)
    
    print("\n" + "=" * 60)
    print("DIGIT RECOGNITION PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': svm_model,
        'test_data': (X_test, y_test),
        'predictions': y_pred,
        'classes': classes,
        'accuracy': SVMEvaluator.accuracy(y_test, y_pred)
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements Support Vector Machine completely from scratch using SMO algorithm.
# 
# Key Dependencies:
# - math: For mathematical operations like exp(), tanh(), sqrt()
# - random: For data generation, shuffling, and SMO optimization
# - typing: For type hints (List, Tuple, Dict, Any, Optional)
# 
# Educational Notes:
# 1. SVM finds optimal hyperplane that maximizes margin between classes
# 2. Kernel trick allows SVM to work in high-dimensional feature spaces
# 3. Sequential Minimal Optimization (SMO) efficiently solves the quadratic optimization problem
# 4. Support vectors are the critical data points that define the decision boundary
# 5. Multi-class classification uses One-vs-Rest strategy
# 6. Different kernels (linear, RBF, polynomial) capture different data patterns
#
# This implementation demonstrates:
# - Complete SVM algorithm with multiple kernel functions
# - SMO optimization algorithm for training efficiency
# - Multi-class classification using binary SVM ensemble
# - Computer vision application with handwritten digit recognition
# - Feature engineering for image data (symmetry, density, center of mass)
# - Comprehensive evaluation with confusion matrix and per-class metrics
# - Synthetic digit generation with noise for realistic training data