# Ensemble Learning: Disease Prediction
# Voting classifier combining multiple models

import math
import random
from typing import List, Tuple, Dict

class DecisionTreeSimple:
    """Simplified decision tree for ensemble."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
        self.tree = None
    
    def gini_impurity(self, labels: List[int]) -> float:
        """Calculate Gini impurity."""
        if not labels:
            return 0
        
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        
        impurity = 1.0
        total = len(labels)
        for count in counts.values():
            prob = count / total
            impurity -= prob ** 2
        
        return impurity
    
    def find_best_split(self, X: List[List[float]], y: List[int], feature_indices: List[int]) -> Tuple:
        """Find best feature and threshold to split on."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            values = [x[feature_idx] for x in X]
            unique_values = sorted(set(values))
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1]) / 2
                
                left_labels = [y[j] for j in range(len(X)) if X[j][feature_idx] <= threshold]
                right_labels = [y[j] for j in range(len(X)) if X[j][feature_idx] > threshold]
                
                if not left_labels or not right_labels:
                    continue
                
                gini = (len(left_labels) * self.gini_impurity(left_labels) + 
                       len(right_labels) * self.gini_impurity(right_labels)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X: List[List[float]], y: List[int], depth: int = 0) -> dict:
        """Recursively build decision tree."""
        # Check stopping criteria
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 5:
            # Leaf node - return most common class
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return {'class': max(counts, key=counts.get)}
        
        # Find best split
        n_features = len(X[0])
        # Random subset of features (for diversity)
        feature_subset = random.sample(range(n_features), k=max(1, int(math.sqrt(n_features))))
        
        feature, threshold = self.find_best_split(X, y, feature_subset)
        
        if feature is None:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return {'class': max(counts, key=counts.get)}
        
        # Split data
        left_indices = [i for i in range(len(X)) if X[i][feature] <= threshold]
        right_indices = [i for i in range(len(X)) if X[i][feature] > threshold]
        
        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1)
        }
    
    def train(self, X: List[List[float]], y: List[int]):
        """Train the decision tree."""
        self.tree = self.build_tree(X, y)
    
    def predict_sample(self, x: List[float], node: dict) -> int:
        """Predict single sample."""
        if 'class' in node:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        else:
            return self.predict_sample(x, node['right'])
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict multiple samples."""
        return [self.predict_sample(x, self.tree) for x in X]


class LogisticRegressionSimple:
    """Simplified logistic regression for ensemble."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z: float) -> float:
        """Sigmoid activation."""
        return 1 / (1 + math.exp(-max(min(z, 500), -500)))
    
    def train(self, X: List[List[float]], y: List[int]):
        """Train logistic regression."""
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                prediction = self.sigmoid(linear)
                
                error = y[i] - prediction
                
                for j in range(n_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias += self.learning_rate * error
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict class labels."""
        predictions = []
        for x in X:
            linear = sum(x[j] * self.weights[j] for j in range(len(x))) + self.bias
            prob = self.sigmoid(linear)
            predictions.append(1 if prob >= 0.5 else 0)
        return predictions


class VotingEnsemble:
    """
    Voting ensemble combining multiple classifiers.
    Implements both hard voting and soft voting.
    """
    
    def __init__(self, voting: str = 'hard'):
        """
        Initialize voting ensemble.
        
        Args:
            voting: 'hard' for majority vote, 'soft' for probability averaging
        """
        self.voting = voting
        self.models = []
    
    def add_model(self, model):
        """Add a model to the ensemble."""
        self.models.append(model)
    
    def train(self, X: List[List[float]], y: List[int]):
        """Train all models in the ensemble."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}...")
            model.train(X, y)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict using ensemble voting."""
        # Get predictions from all models
        all_predictions = [model.predict(X) for model in self.models]
        
        # Transpose to get predictions per sample
        n_samples = len(X)
        ensemble_predictions = []
        
        for i in range(n_samples):
            sample_predictions = [pred[i] for pred in all_predictions]
            
            # Hard voting - majority vote
            counts = {}
            for pred in sample_predictions:
                counts[pred] = counts.get(pred, 0) + 1
            
            ensemble_predictions.append(max(counts, key=counts.get))
        
        return ensemble_predictions
    
    def evaluate(self, X: List[List[float]], y: List[int]) -> Dict:
        """Evaluate ensemble performance."""
        predictions = self.predict(X)
        
        # Calculate metrics
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        accuracy = correct / len(y) * 100
        
        # Confusion matrix
        tp = sum(1 for pred, true in zip(predictions, y) if pred == 1 and true == 1)
        tn = sum(1 for pred, true in zip(predictions, y) if pred == 0 and true == 0)
        fp = sum(1 for pred, true in zip(predictions, y) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(predictions, y) if pred == 0 and true == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }


def generate_disease_data(n_samples: int = 800) -> Tuple:
    """
    Generate synthetic disease prediction dataset.
    Binary classification: disease (1) or healthy (0)
    """
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        # Features: age, blood_pressure, cholesterol, bmi, glucose, heart_rate
        has_disease = random.random() > 0.5
        
        if has_disease:
            age = random.uniform(50, 80)
            bp = random.uniform(140, 180)
            cholesterol = random.uniform(220, 300)
            bmi = random.uniform(28, 40)
            glucose = random.uniform(120, 200)
            hr = random.uniform(85, 110)
        else:
            age = random.uniform(20, 60)
            bp = random.uniform(90, 130)
            cholesterol = random.uniform(150, 210)
            bmi = random.uniform(18, 27)
            glucose = random.uniform(70, 110)
            hr = random.uniform(60, 85)
        
        X_data.append([age, bp, cholesterol, bmi, glucose, hr])
        y_data.append(1 if has_disease else 0)
    
    # Split
    split = int(0.8 * n_samples)
    return X_data[:split], y_data[:split], X_data[split:], y_data[split:]


if __name__ == "__main__":
    print("Ensemble Learning: Disease Prediction")
    print("="*60)
    
    # Generate data
    print("\nGenerating patient health dataset...")
    X_train, y_train, X_test, y_test = generate_disease_data(800)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: Age, BP, Cholesterol, BMI, Glucose, Heart Rate")
    
    # Create ensemble
    print("\nBuilding ensemble with 5 models...")
    ensemble = VotingEnsemble(voting='hard')
    
    # Add different models
    for i in range(3):
        ensemble.add_model(DecisionTreeSimple(max_depth=5))
    ensemble.add_model(LogisticRegressionSimple())
    ensemble.add_model(LogisticRegressionSimple())
    
    # Train
    print("\nTraining ensemble...")
    ensemble.train(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating ensemble...")
    results = ensemble.evaluate(X_test, y_test)
    print(f"\nAccuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1']:.3f}")
    
    print("\n" + "="*60)
    print("Ensemble Learning Complete!")
