# Linear Regression: Student Performance Analysis
# CBSE Class 12 AI - Supervised Learning Project
# This project implements linear regression from scratch to predict student exam scores

import math
import random
from typing import List, Tuple, Dict

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("LINEAR REGRESSION: STUDENT PERFORMANCE ANALYSIS")
    print("CBSE Class 12 AI - Supervised Learning Project")
    print("=" * 80)
    print()

class LinearRegression:
    """Complete Linear Regression implementation from scratch"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.learning_rate = 0.01
        self.iterations = 1000
        
    def add_polynomial_features(self, X: List[List[float]], degree: int = 2) -> List[List[float]]:
        """Add polynomial features to the dataset"""
        X_poly = []
        for row in X:
            new_row = row.copy()
            # Add polynomial features
            for d in range(2, degree + 1):
                for i in range(len(row)):
                    new_row.append(row[i] ** d)
            # Add interaction terms for 2 features
            if len(row) >= 2:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        new_row.append(row[i] * row[j])
            X_poly.append(new_row)
        return X_poly
    
    def normalize_features(self, X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
        """Normalize features using Z-score normalization"""
        if not X or not X[0]:
            return X, [], []
            
        n_features = len(X[0])
        means = []
        stds = []
        
        # Calculate means
        for j in range(n_features):
            feature_values = [X[i][j] for i in range(len(X))]
            mean = sum(feature_values) / len(feature_values)
            means.append(mean)
        
        # Calculate standard deviations
        for j in range(n_features):
            feature_values = [X[i][j] for i in range(len(X))]
            variance = sum([(x - means[j]) ** 2 for x in feature_values]) / len(feature_values)
            std = math.sqrt(variance)
            stds.append(std if std > 0 else 1)  # Prevent division by zero
        
        # Normalize
        X_normalized = []
        for i in range(len(X)):
            normalized_row = []
            for j in range(n_features):
                normalized_value = (X[i][j] - means[j]) / stds[j]
                normalized_row.append(normalized_value)
            X_normalized.append(normalized_row)
            
        return X_normalized, means, stds
    
    def fit(self, X: List[List[float]], y: List[float], learning_rate: float = 0.01, iterations: int = 1000):
        """Train the linear regression model using gradient descent"""
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        # Normalize features
        X_normalized, self.feature_means, self.feature_stds = self.normalize_features(X)
        
        n_samples = len(X_normalized)
        n_features = len(X_normalized[0]) if X_normalized else 0
        
        # Initialize weights and bias
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        self.cost_history = []
        
        # Gradient descent
        for iteration in range(iterations):
            # Forward pass
            y_predicted = []
            for i in range(n_samples):
                prediction = self.bias
                for j in range(n_features):
                    prediction += self.weights[j] * X_normalized[i][j]
                y_predicted.append(prediction)
            
            # Calculate cost (Mean Squared Error)
            cost = sum([(y_predicted[i] - y[i]) ** 2 for i in range(n_samples)]) / (2 * n_samples)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_predicted[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X_normalized[i][j]
            
            # Update weights and bias
            for j in range(n_features):
                self.weights[j] -= learning_rate * (dw[j] / n_samples)
            self.bias -= learning_rate * (db / n_samples)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration:4d}: Cost = {cost:.6f}")
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions on new data"""
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Normalize using training statistics
        X_normalized = []
        for i in range(len(X)):
            normalized_row = []
            for j in range(len(X[0])):
                if j < len(self.feature_means):
                    normalized_value = (X[i][j] - self.feature_means[j]) / self.feature_stds[j]
                    normalized_row.append(normalized_value)
            X_normalized.append(normalized_row)
        
        predictions = []
        for row in X_normalized:
            prediction = self.bias
            for j in range(len(row)):
                if j < len(self.weights):
                    prediction += self.weights[j] * row[j]
            predictions.append(prediction)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Return model summary statistics"""
        return {
            'weights': self.weights.copy() if self.weights else None,
            'bias': self.bias,
            'n_features': len(self.weights) if self.weights else 0,
            'final_cost': self.cost_history[-1] if self.cost_history else None,
            'iterations_trained': len(self.cost_history)
        }

class ModelEvaluator:
    """Comprehensive model evaluation metrics"""
    
    @staticmethod
    def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Squared Error"""
        return sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))]) / len(y_true)
    
    @staticmethod
    def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        return sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]) / len(y_true)
    
    @staticmethod
    def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Root Mean Squared Error"""
        mse = ModelEvaluator.mean_squared_error(y_true, y_pred)
        return math.sqrt(mse)
    
    @staticmethod
    def r_squared(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate R-squared (coefficient of determination)"""
        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum([(y - y_mean) ** 2 for y in y_true])
        ss_res = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def adjusted_r_squared(y_true: List[float], y_pred: List[float], n_features: int) -> float:
        """Calculate Adjusted R-squared"""
        r2 = ModelEvaluator.r_squared(y_true, y_pred)
        n = len(y_true)
        
        if n <= n_features + 1:
            return r2
        
        return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

class StudentDataGenerator:
    """Generate realistic student performance data"""
    
    @staticmethod
    def generate_student_data(n_students: int = 200) -> Tuple[List[List[float]], List[float], List[str]]:
        """Generate synthetic student performance dataset"""
        random.seed(42)  # For reproducible results
        
        students_data = []
        exam_scores = []
        feature_names = ['study_hours_per_day', 'previous_grade', 'attendance_rate', 
                        'sleep_hours', 'assignments_completed', 'extra_curricular_hours']
        
        for i in range(n_students):
            # Generate correlated features that influence exam performance
            study_hours = random.uniform(1.0, 8.0)  # Hours per day
            previous_grade = random.uniform(60.0, 95.0)  # Previous academic performance
            attendance_rate = random.uniform(70.0, 98.0)  # Percentage
            sleep_hours = random.uniform(5.0, 9.0)  # Hours per night
            assignments_completed = random.uniform(60.0, 100.0)  # Percentage
            extra_curricular = random.uniform(0.0, 4.0)  # Hours per week
            
            # Create realistic relationships
            # More study hours generally lead to better scores
            score_from_study = study_hours * 8.5
            
            # Previous performance is a strong predictor
            score_from_previous = previous_grade * 0.4
            
            # Attendance affects performance
            score_from_attendance = (attendance_rate - 70) * 0.3
            
            # Adequate sleep helps (optimal around 7-8 hours)
            optimal_sleep = 7.5
            sleep_effect = 10 - abs(sleep_hours - optimal_sleep) * 3
            
            # Assignment completion matters
            score_from_assignments = assignments_completed * 0.15
            
            # Too much extra-curricular can hurt (balance is key)
            if extra_curricular > 2.5:
                extra_effect = -(extra_curricular - 2.5) * 2
            else:
                extra_effect = extra_curricular * 1.5
            
            # Combine effects with some noise
            base_score = (score_from_study + score_from_previous + 
                         score_from_attendance + sleep_effect + 
                         score_from_assignments + extra_effect)
            
            # Add random noise
            noise = random.uniform(-8, 8)
            final_score = max(0, min(100, base_score + noise))
            
            students_data.append([
                study_hours, previous_grade, attendance_rate, 
                sleep_hours, assignments_completed, extra_curricular
            ])
            exam_scores.append(final_score)
        
        return students_data, exam_scores, feature_names

def split_data(X: List[List[float]], y: List[float], train_ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
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

def print_data_analysis(X: List[List[float]], y: List[float], feature_names: List[str]):
    """Print comprehensive data analysis"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    print(f"Number of students: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Features: {', '.join(feature_names)}")
    
    # Feature statistics
    print("\nFeature Statistics:")
    print("-" * 40)
    for j in range(n_features):
        feature_values = [X[i][j] for i in range(n_samples)]
        mean_val = sum(feature_values) / len(feature_values)
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        print(f"{feature_names[j]:25s}: Mean={mean_val:6.2f}, Min={min_val:6.2f}, Max={max_val:6.2f}")
    
    # Target variable statistics
    y_mean = sum(y) / len(y)
    y_min = min(y)
    y_max = max(y)
    print(f"\nExam Score Statistics:")
    print(f"Mean: {y_mean:.2f}, Min: {y_min:.2f}, Max: {y_max:.2f}")

def print_model_results(model: LinearRegression, X_test: List[List[float]], y_test: List[float], feature_names: List[str]):
    """Print comprehensive model evaluation results"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    evaluator = ModelEvaluator()
    mse = evaluator.mean_squared_error(y_test, y_pred)
    mae = evaluator.mean_absolute_error(y_test, y_pred)
    rmse = evaluator.root_mean_squared_error(y_test, y_pred)
    r2 = evaluator.r_squared(y_test, y_pred)
    adj_r2 = evaluator.adjusted_r_squared(y_test, y_pred, len(feature_names))
    
    print(f"Mean Squared Error (MSE):     {mse:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"Root Mean Squared Error:      {rmse:.4f}")
    print(f"R-squared (R²):              {r2:.4f}")
    print(f"Adjusted R-squared:          {adj_r2:.4f}")
    
    # Model summary
    summary = model.get_model_summary()
    print(f"\nModel Parameters:")
    print(f"Bias (intercept): {summary['bias']:.4f}")
    print(f"\nFeature Weights:")
    for i, (feature, weight) in enumerate(zip(feature_names, summary['weights'][:len(feature_names)])):
        print(f"{feature:25s}: {weight:8.4f}")
    
    # Sample predictions
    print(f"\nSample Predictions (First 10 test cases):")
    print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 35)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_pred[i])
        print(f"{y_test[i]:<10.2f} {y_pred[i]:<10.2f} {error:<10.2f}")

# Main execution function
def main():
    """Main function to demonstrate linear regression on student performance data"""
    print_header()
    
    # Generate student performance dataset
    print("Generating student performance dataset...")
    X, y, feature_names = StudentDataGenerator.generate_student_data(n_students=200)
    
    # Analyze the dataset
    print_data_analysis(X, y, feature_names)
    
    # Split data into training and testing sets
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    print(f"Training set: {len(X_train)} students")
    print(f"Testing set: {len(X_test)} students")
    
    # Create and train the model
    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 60)
    
    # Add polynomial features for better performance
    print("\nAdding polynomial features...")
    model = LinearRegression()
    X_train_poly = model.add_polynomial_features(X_train, degree=2)
    X_test_poly = model.add_polynomial_features(X_test, degree=2)
    
    # Train the model
    print(f"\nTraining model with {len(X_train_poly[0])} features (including polynomial terms)...")
    model.fit(X_train_poly, y_train, learning_rate=0.01, iterations=1500)
    
    # Evaluate the model
    print_model_results(model, X_test_poly, y_test, feature_names)
    
    # Feature importance analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    weights = model.get_model_summary()['weights']
    feature_importance = [(abs(weights[i]), feature_names[i]) for i in range(min(len(weights), len(feature_names)))]
    feature_importance.sort(reverse=True)
    
    print("\nTop 5 Most Important Features:")
    for i, (importance, feature) in enumerate(feature_importance[:5]):
        print(f"{i+1}. {feature:25s}: {importance:.4f}")
    
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': model,
        'test_data': (X_test_poly, y_test),
        'metrics': {
            'mse': ModelEvaluator.mean_squared_error(y_test, model.predict(X_test_poly)),
            'r2': ModelEvaluator.r_squared(y_test, model.predict(X_test_poly))
        },
        'feature_names': feature_names
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements linear regression completely from scratch using only Python's built-in libraries.
# 
# Key Dependencies:
# - math: For mathematical operations like sqrt()
# - random: For data generation and train/test splitting
# - typing: For type hints (List, Tuple, Dict)
# 
# Educational Notes:
# 1. Linear regression finds the best-fit line through data points
# 2. Gradient descent optimizes the model by minimizing the cost function
# 3. Feature normalization helps gradient descent converge faster
# 4. Polynomial features can capture non-linear relationships
# 5. R-squared measures how well the model explains the variance in data
# 6. Cross-validation would be the next step for more robust evaluation
#
# This implementation demonstrates core machine learning concepts:
# - Supervised learning with continuous target variables
# - Feature engineering (polynomial features, normalization)
# - Model training using gradient descent optimization
# - Comprehensive model evaluation with multiple metrics
# - Real-world application to educational data analysis