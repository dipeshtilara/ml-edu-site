# Decision Tree: Medical Diagnosis System
# CBSE Class 12 AI - Tree-based Learning Project
# This project implements decision tree from scratch for medical diagnosis

import math
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("DECISION TREE: MEDICAL DIAGNOSIS SYSTEM")
    print("CBSE Class 12 AI - Tree-based Learning Project")
    print("=" * 80)
    print()

class TreeNode:
    """Node class for decision tree structure"""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, samples=0):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold          # Threshold value for split
        self.left = left                   # Left child node
        self.right = right                 # Right child node
        self.value = value                 # Class prediction for leaf nodes
        self.samples = samples             # Number of samples in this node
        self.impurity = 0.0               # Impurity measure for this node
        self.gain = 0.0                   # Information gain from this split

class DecisionTree:
    """Complete Decision Tree implementation from scratch"""
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.feature_names = None
        self.class_names = None
        self.n_features = 0
        self.n_classes = 0
    
    def calculate_gini(self, y: List[int]) -> float:
        """Calculate Gini impurity"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        gini = 1.0
        for count in class_counts.values():
            probability = count / total_samples
            gini -= probability ** 2
        
        return gini
    
    def calculate_entropy(self, y: List[int]) -> float:
        """Calculate entropy (information)"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                probability = count / total_samples
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_impurity(self, y: List[int]) -> float:
        """Calculate impurity based on chosen criterion"""
        if self.criterion == 'gini':
            return self.calculate_gini(y)
        elif self.criterion == 'entropy':
            return self.calculate_entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def find_best_split(self, X: List[List[float]], y: List[int]) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best feature and threshold to split on"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        if n_samples <= 1:
            return None, None, 0.0
        
        # Calculate parent impurity
        parent_impurity = self.calculate_impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature
            feature_values = sorted(set(X[i][feature_idx] for i in range(n_samples)))
            
            # Try each unique value as a threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_indices = []
                right_indices = []
                
                for j in range(n_samples):
                    if X[j][feature_idx] <= threshold:
                        left_indices.append(j)
                    else:
                        right_indices.append(j)
                
                # Skip if split doesn't meet minimum sample requirements
                if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity after split
                left_y = [y[j] for j in left_indices]
                right_y = [y[j] for j in right_indices]
                
                left_impurity = self.calculate_impurity(left_y)
                right_impurity = self.calculate_impurity(right_y)
                
                weighted_impurity = (len(left_y) * left_impurity + len(right_y) * right_impurity) / n_samples
                
                # Calculate information gain
                gain = parent_impurity - weighted_impurity
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X: List[List[float]], y: List[int], depth: int = 0) -> TreeNode:
        """Recursively build the decision tree"""
        n_samples = len(X)
        
        # Create node
        node = TreeNode(samples=n_samples)
        node.impurity = self.calculate_impurity(y)
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(set(y)) == 1 or
            node.impurity == 0):
            
            # Make this a leaf node
            class_counts = Counter(y)
            node.value = max(class_counts.items(), key=lambda x: x[1])[0]
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            # Make this a leaf node
            class_counts = Counter(y)
            node.value = max(class_counts.items(), key=lambda x: x[1])[0]
            return node
        
        # Set split parameters
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        # Split data
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(n_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        # Recursively build subtrees
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)
        
        return node
    
    def fit(self, X: List[List[float]], y: List[int], feature_names: List[str] = None, class_names: List[str] = None):
        """Train the decision tree"""
        self.n_features = len(X[0]) if X else 0
        self.n_classes = len(set(y))
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.n_features)]
        self.class_names = class_names or [f"Class_{i}" for i in range(self.n_classes)]
        
        print(f"Building decision tree with {len(X)} samples and {self.n_features} features...")
        self.root = self.build_tree(X, y)
        print(f"Tree construction completed!")
    
    def predict_sample(self, sample: List[float], node: TreeNode = None) -> int:
        """Predict class for a single sample"""
        if node is None:
            node = self.root
        
        # If leaf node, return the class
        if node.value is not None:
            return node.value
        
        # Navigate to appropriate child
        if sample[node.feature_index] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Make predictions for multiple samples"""
        if self.root is None:
            raise ValueError("Tree has not been trained yet. Call fit() first.")
        
        return [self.predict_sample(sample) for sample in X]
    
    def print_tree(self, node: TreeNode = None, depth: int = 0, prefix: str = "Root: "):
        """Print tree structure in a readable format"""
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.value is not None:
            # Leaf node
            class_name = self.class_names[node.value] if node.value < len(self.class_names) else f"Class_{node.value}"
            print(f"{indent}{prefix}Predict: {class_name} (samples: {node.samples}, impurity: {node.impurity:.3f})")
        else:
            # Internal node
            feature_name = self.feature_names[node.feature_index]
            print(f"{indent}{prefix}{feature_name} <= {node.threshold:.3f} (gain: {node.gain:.3f}, samples: {node.samples})")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")
    
    def get_tree_depth(self, node: TreeNode = None) -> int:
        """Calculate the depth of the tree"""
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            return 1
        
        left_depth = self.get_tree_depth(node.left) if node.left else 0
        right_depth = self.get_tree_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self, node: TreeNode = None) -> int:
        """Count total number of nodes in the tree"""
        if node is None:
            node = self.root
        
        if node.value is not None:  # Leaf node
            return 1
        
        left_count = self.count_nodes(node.left) if node.left else 0
        right_count = self.count_nodes(node.right) if node.right else 0
        
        return 1 + left_count + right_count

class MedicalDataGenerator:
    """Generate realistic medical diagnosis dataset"""
    
    @staticmethod
    def generate_medical_data(n_patients: int = 500) -> Tuple[List[List[float]], List[int], List[str], List[str]]:
        """Generate synthetic medical diagnosis dataset"""
        random.seed(42)  # For reproducible results
        
        feature_names = [
            'age', 'temperature', 'heart_rate', 'blood_pressure_systolic', 
            'blood_pressure_diastolic', 'white_blood_cell_count', 
            'red_blood_cell_count', 'hemoglobin_level'
        ]
        
        # Disease classes: 0=Healthy, 1=Flu, 2=Hypertension, 3=Anemia, 4=Infection
        class_names = ['Healthy', 'Flu', 'Hypertension', 'Anemia', 'Infection']
        
        patients_data = []
        diagnoses = []
        
        for _ in range(n_patients):
            # Generate patient characteristics
            age = random.uniform(18, 80)
            
            # Randomly select a condition with different probabilities
            condition_prob = random.random()
            
            if condition_prob < 0.3:  # Healthy (30%)
                condition = 0
                temperature = random.uniform(97.5, 99.5)  # Normal temperature
                heart_rate = random.uniform(60, 100)      # Normal heart rate
                bp_systolic = random.uniform(110, 140)    # Normal blood pressure
                bp_diastolic = random.uniform(70, 90)
                wbc_count = random.uniform(4000, 11000)   # Normal WBC count
                rbc_count = random.uniform(4.5, 5.5)      # Normal RBC count
                hemoglobin = random.uniform(12, 16)       # Normal hemoglobin
                
            elif condition_prob < 0.5:  # Flu (20%)
                condition = 1
                temperature = random.uniform(100, 104)     # Fever
                heart_rate = random.uniform(90, 120)      # Elevated heart rate
                bp_systolic = random.uniform(100, 130)
                bp_diastolic = random.uniform(60, 85)
                wbc_count = random.uniform(3000, 8000)    # Slightly low WBC
                rbc_count = random.uniform(4.2, 5.2)
                hemoglobin = random.uniform(11, 15)
                
            elif condition_prob < 0.7:  # Hypertension (20%)
                condition = 2
                temperature = random.uniform(97, 99)
                heart_rate = random.uniform(70, 110)
                bp_systolic = random.uniform(140, 180)    # High blood pressure
                bp_diastolic = random.uniform(90, 110)    # High diastolic
                wbc_count = random.uniform(5000, 12000)
                rbc_count = random.uniform(4.3, 5.3)
                hemoglobin = random.uniform(12, 16)
                
            elif condition_prob < 0.85:  # Anemia (15%)
                condition = 3
                temperature = random.uniform(97, 99.5)
                heart_rate = random.uniform(80, 120)      # May be elevated
                bp_systolic = random.uniform(90, 120)     # May be low
                bp_diastolic = random.uniform(50, 80)
                wbc_count = random.uniform(4000, 10000)
                rbc_count = random.uniform(3.5, 4.5)      # Low RBC count
                hemoglobin = random.uniform(7, 11)        # Low hemoglobin
                
            else:  # Infection (15%)
                condition = 4
                temperature = random.uniform(100.5, 105)  # High fever
                heart_rate = random.uniform(100, 140)     # Elevated heart rate
                bp_systolic = random.uniform(90, 130)
                bp_diastolic = random.uniform(50, 85)
                wbc_count = random.uniform(12000, 20000)  # Elevated WBC
                rbc_count = random.uniform(4.0, 5.0)
                hemoglobin = random.uniform(10, 14)
            
            # Add some age-related variations
            if age > 60:
                bp_systolic += random.uniform(5, 15)  # Older patients tend to have higher BP
                heart_rate -= random.uniform(5, 10)   # Slower resting heart rate
            
            # Add small amount of noise to make data realistic
            noise_factor = 0.05
            temperature += random.uniform(-noise_factor, noise_factor)
            heart_rate += random.uniform(-noise_factor * 10, noise_factor * 10)
            bp_systolic += random.uniform(-noise_factor * 20, noise_factor * 20)
            bp_diastolic += random.uniform(-noise_factor * 10, noise_factor * 10)
            wbc_count += random.uniform(-noise_factor * 1000, noise_factor * 1000)
            rbc_count += random.uniform(-noise_factor, noise_factor)
            hemoglobin += random.uniform(-noise_factor * 2, noise_factor * 2)
            
            # Ensure values stay within realistic ranges
            temperature = max(95, min(108, temperature))
            heart_rate = max(40, min(200, heart_rate))
            bp_systolic = max(80, min(220, bp_systolic))
            bp_diastolic = max(40, min(120, bp_diastolic))
            wbc_count = max(1000, min(30000, wbc_count))
            rbc_count = max(2.0, min(7.0, rbc_count))
            hemoglobin = max(6, min(20, hemoglobin))
            
            patient_features = [
                age, temperature, heart_rate, bp_systolic, 
                bp_diastolic, wbc_count, rbc_count, hemoglobin
            ]
            
            patients_data.append(patient_features)
            diagnoses.append(condition)
        
        return patients_data, diagnoses, feature_names, class_names

class TreeEvaluator:
    """Comprehensive evaluation metrics for decision trees"""
    
    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy"""
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        return correct / len(y_true)
    
    @staticmethod
    def confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> List[List[int]]:
        """Calculate confusion matrix"""
        matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
        
        for i in range(len(y_true)):
            matrix[y_true[i]][y_pred[i]] += 1
        
        return matrix
    
    @staticmethod
    def classification_report(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict:
        """Generate detailed classification report"""
        n_classes = len(class_names)
        confusion_mat = TreeEvaluator.confusion_matrix(y_true, y_pred, n_classes)
        
        report = {}
        
        for i in range(n_classes):
            tp = confusion_mat[i][i]
            fp = sum(confusion_mat[j][i] for j in range(n_classes)) - tp
            fn = sum(confusion_mat[i][j] for j in range(n_classes)) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report[class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': sum(1 for label in y_true if label == i)
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

def print_data_analysis(X: List[List[float]], y: List[int], feature_names: List[str], class_names: List[str]):
    """Print comprehensive data analysis"""
    print("\n" + "=" * 60)
    print("MEDICAL DATASET ANALYSIS")
    print("=" * 60)
    
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    print(f"Number of patients: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Medical features: {', '.join(feature_names)}")
    
    # Class distribution
    class_counts = Counter(y)
    print(f"\nDiagnosis Distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id]
        percentage = count / n_samples * 100
        print(f"{class_name:15s}: {count:3d} patients ({percentage:5.1f}%)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"{'Feature':<25} {'Mean':<8} {'Min':<8} {'Max':<8}")
    print("-" * 55)
    for j in range(n_features):
        feature_values = [X[i][j] for i in range(n_samples)]
        mean_val = sum(feature_values) / len(feature_values)
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        print(f"{feature_names[j]:<25} {mean_val:<8.2f} {min_val:<8.2f} {max_val:<8.2f}")

def print_evaluation_results(y_true: List[int], y_pred: List[int], class_names: List[str], tree: DecisionTree):
    """Print comprehensive evaluation results"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = TreeEvaluator.accuracy(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Tree statistics
    tree_depth = tree.get_tree_depth()
    node_count = tree.count_nodes()
    print(f"\nTree Statistics:")
    print(f"Tree Depth: {tree_depth}")
    print(f"Total Nodes: {node_count}")
    print(f"Leaf Nodes: {node_count - (node_count - 1) // 2 if node_count > 1 else 1}")
    
    # Confusion matrix
    confusion_mat = TreeEvaluator.confusion_matrix(y_true, y_pred, len(class_names))
    print(f"\nConfusion Matrix:")
    print(f"{'Actual\\Predicted':<15}", end="")
    for name in class_names:
        print(f"{name:<10}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:<15}", end="")
        for j in range(len(class_names)):
            print(f"{confusion_mat[i][j]:<10}", end="")
        print()
    
    # Classification report
    report = TreeEvaluator.classification_report(y_true, y_pred, class_names)
    print(f"\nClassification Report:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    
    for class_name, metrics in report.items():
        print(f"{class_name:<15} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
              f"{metrics['f1_score']:<10.3f} {metrics['support']:<10d}")

def main():
    """Main function to demonstrate decision tree for medical diagnosis"""
    print_header()
    
    # Generate medical dataset
    print("Generating medical diagnosis dataset...")
    X, y, feature_names, class_names = MedicalDataGenerator.generate_medical_data(n_patients=500)
    
    # Analyze dataset
    print_data_analysis(X, y, feature_names, class_names)
    
    # Split data
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    print(f"Training set: {len(X_train)} patients")
    print(f"Testing set: {len(X_test)} patients")
    
    # Create and train decision tree
    print("\n" + "=" * 60)
    print("TRAINING DECISION TREE MODEL")
    print("=" * 60)
    
    tree = DecisionTree(max_depth=8, min_samples_split=5, min_samples_leaf=2, criterion='gini')
    tree.fit(X_train, y_train, feature_names, class_names)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = tree.predict(X_test)
    
    # Evaluate model
    print_evaluation_results(y_test, y_pred, class_names, tree)
    
    # Print tree structure
    print("\n" + "=" * 60)
    print("DECISION TREE STRUCTURE")
    print("=" * 60)
    print("\nTree Structure (showing decision rules):")
    tree.print_tree()
    
    print("\n" + "=" * 60)
    print("MEDICAL DIAGNOSIS PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': tree,
        'test_data': (X_test, y_test),
        'predictions': y_pred,
        'feature_names': feature_names,
        'class_names': class_names,
        'accuracy': TreeEvaluator.accuracy(y_test, y_pred)
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements decision tree completely from scratch using only Python built-ins.
# 
# Key Dependencies:
# - math: For mathematical operations like log2() for entropy calculation
# - random: For data generation and train/test splitting
# - typing: For type hints (List, Tuple, Dict, Any, Optional)
# - collections.Counter: For counting class occurrences
# 
# Educational Notes:
# 1. Decision trees use recursive binary splits to partition the data
# 2. Gini impurity and entropy are common measures of node "purity"
# 3. Information gain guides the selection of optimal splits
# 4. Tree pruning (controlled by hyperparameters) prevents overfitting
# 5. Decision trees are interpretable - you can follow the decision path
# 6. Trees can handle both numerical and categorical features naturally
#
# This implementation demonstrates:
# - Complete decision tree algorithm with multiple splitting criteria
# - Medical diagnosis as a real-world multi-class classification problem
# - Tree visualization and interpretation for explainable AI
# - Comprehensive evaluation with confusion matrix and per-class metrics
# - Hyperparameter tuning for tree depth and minimum samples
# - Synthetic medical data generation with realistic feature relationships