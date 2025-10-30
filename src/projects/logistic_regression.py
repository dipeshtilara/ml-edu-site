# Logistic Regression: Email Spam Detection
# CBSE Class 12 AI - Binary Classification Project
# This project implements logistic regression from scratch for email spam detection

import math
import random
from typing import List, Tuple, Dict, Set
import re

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("LOGISTIC REGRESSION: EMAIL SPAM DETECTION")
    print("CBSE Class 12 AI - Binary Classification Project")
    print("=" * 80)
    print()

class TextPreprocessor:
    """Comprehensive text preprocessing for email analysis"""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had',
            'what', 'said', 'each', 'which', 'do', 'their', 'time', 'if', 'up',
            'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
            'make', 'like', 'into', 'him', 'two', 'more', 'very', 'after', 'back',
            'other', 'many', 'than', 'first', 'been', 'who', 'its', 'now', 'people',
            'my', 'made', 'over', 'did', 'down', 'only', 'way', 'find', 'use', 'may',
            'water', 'long', 'little', 'very', 'after', 'words', 'called'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' url ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' email ', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' phone ', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        
        # Remove stop words and short words
        tokens = [token for token in tokens if len(token) > 2 and token not in self.stop_words]
        
        return tokens
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive features from text"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        
        # Count specific patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_count'] = sum(1 for c in text if c.isupper())
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        
        # Spam indicators
        spam_keywords = ['free', 'money', 'win', 'winner', 'cash', 'prize', 'offer', 
                        'deal', 'click', 'buy', 'sale', 'discount', 'urgent', 'limited',
                        'guarantee', 'risk', 'investment', 'loan', 'debt']
        
        text_lower = text.lower()
        for keyword in spam_keywords:
            features[f'has_{keyword}'] = 1.0 if keyword in text_lower else 0.0
        
        # URL and email presence
        features['has_url'] = 1.0 if 'http' in text_lower else 0.0
        features['has_email'] = 1.0 if '@' in text else 0.0
        
        return features

class TFIDFVectorizer:
    """TF-IDF (Term Frequency - Inverse Document Frequency) implementation"""
    
    def __init__(self, max_features: int = 1000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary = {}
        self.idf_values = {}
        self.preprocessor = TextPreprocessor()
    
    def fit(self, documents: List[str]):
        """Learn vocabulary and IDF values from documents"""
        # Tokenize all documents
        tokenized_docs = [self.preprocessor.tokenize(doc) for doc in documents]
        
        # Count document frequency for each term
        term_doc_count = {}
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] = term_doc_count.get(token, 0) + 1
        
        # Filter terms by minimum document frequency
        filtered_terms = {term: count for term, count in term_doc_count.items() 
                         if count >= self.min_df}
        
        # Select top features by document frequency
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        selected_terms = sorted_terms[:self.max_features]
        
        # Create vocabulary
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(selected_terms)}
        
        # Calculate IDF values
        n_docs = len(documents)
        for term in self.vocabulary:
            df = term_doc_count[term]
            self.idf_values[term] = math.log(n_docs / df)
    
    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors"""
        tfidf_vectors = []
        
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc)
            
            # Calculate term frequencies
            term_freq = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Normalize term frequencies
            max_freq = max(term_freq.values()) if term_freq else 1
            normalized_tf = {term: freq / max_freq for term, freq in term_freq.items()}
            
            # Create TF-IDF vector
            tfidf_vector = [0.0] * len(self.vocabulary)
            for term, tf in normalized_tf.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    idf = self.idf_values[term]
                    tfidf_vector[idx] = tf * idf
            
            tfidf_vectors.append(tfidf_vector)
        
        return tfidf_vectors
    
    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)

class LogisticRegression:
    """Complete Logistic Regression implementation from scratch"""
    
    def __init__(self):
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.learning_rate = 0.01
        self.iterations = 1000
    
    def sigmoid(self, z: float) -> float:
        """Sigmoid activation function with numerical stability"""
        if z > 500:  # Prevent overflow
            return 1.0
        elif z < -500:  # Prevent underflow
            return 0.0
        else:
            return 1.0 / (1.0 + math.exp(-z))
    
    def fit(self, X: List[List[float]], y: List[int], learning_rate: float = 0.01, iterations: int = 1000):
        """Train logistic regression using gradient descent"""
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Initialize parameters
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        self.cost_history = []
        
        # Gradient descent
        for iteration in range(iterations):
            # Forward pass
            predictions = []
            for i in range(n_samples):
                z = self.bias
                for j in range(n_features):
                    z += self.weights[j] * X[i][j]
                predictions.append(self.sigmoid(z))
            
            # Calculate cost (cross-entropy)
            cost = 0.0
            for i in range(n_samples):
                y_pred = max(1e-15, min(1 - 1e-15, predictions[i]))  # Clip to prevent log(0)
                cost += -(y[i] * math.log(y_pred) + (1 - y[i]) * math.log(1 - y_pred))
            cost /= n_samples
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = predictions[i] - y[i]
                db += error
                for j in range(n_features):
                    dw[j] += error * X[i][j]
            
            # Update parameters
            for j in range(n_features):
                self.weights[j] -= learning_rate * (dw[j] / n_samples)
            self.bias -= learning_rate * (db / n_samples)
            
            # Print progress
            if iteration % 200 == 0:
                print(f"Iteration {iteration:4d}: Cost = {cost:.6f}")
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict class probabilities"""
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        probabilities = []
        for row in X:
            z = self.bias
            for j in range(len(row)):
                if j < len(self.weights):
                    z += self.weights[j] * row[j]
            probabilities.append(self.sigmoid(z))
        
        return probabilities
    
    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return [1 if prob >= threshold else 0 for prob in probabilities]

class ClassificationEvaluator:
    """Comprehensive classification evaluation metrics"""
    
    @staticmethod
    def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix elements"""
        tp = fp = tn = fn = 0
        
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
        
        return tp, fp, tn, fn
    
    @staticmethod
    def accuracy(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate accuracy"""
        correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
        return correct / len(y_true)
    
    @staticmethod
    def precision(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate precision"""
        tp, fp, _, _ = ClassificationEvaluator.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def recall(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate recall (sensitivity)"""
        tp, _, _, fn = ClassificationEvaluator.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    @staticmethod
    def f1_score(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate F1 score"""
        prec = ClassificationEvaluator.precision(y_true, y_pred)
        rec = ClassificationEvaluator.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    @staticmethod
    def specificity(y_true: List[int], y_pred: List[int]) -> float:
        """Calculate specificity (true negative rate)"""
        _, fp, tn, _ = ClassificationEvaluator.confusion_matrix(y_true, y_pred)
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

class EmailDataGenerator:
    """Generate realistic email dataset for spam detection"""
    
    @staticmethod
    def generate_ham_emails(n_emails: int) -> List[str]:
        """Generate legitimate (ham) email content"""
        ham_templates = [
            "Hi {name}, I hope you are doing well. I wanted to follow up on our meeting yesterday about the {topic} project. Please let me know your thoughts on the proposal we discussed. Looking forward to hearing from you soon. Best regards, {sender}",
            
            "Dear {name}, Thank you for your email regarding the {topic}. I have reviewed the documents you sent and I think we can proceed with the next steps. Could we schedule a call next week to discuss the details? Please let me know what time works best for you. Kind regards, {sender}",
            
            "Hello {name}, I hope your week is going well. I wanted to share some updates about the {topic} we have been working on. The team has made good progress and we should be ready for the next phase soon. I will keep you updated on our progress. Have a great day! {sender}",
            
            "Hi {name}, I received your message about the {topic}. This sounds like an interesting opportunity and I would like to learn more about it. Could you provide additional information about the requirements and timeline? I look forward to discussing this further. Best, {sender}",
            
            "Dear {name}, I wanted to reach out to you about the upcoming {topic} event. We have received positive feedback from the participants and everything is on track. Please let me know if you have any questions or concerns. Thank you for your continued support. Sincerely, {sender}"
        ]
        
        names = ['John', 'Sarah', 'Michael', 'Lisa', 'David', 'Emily', 'Robert', 'Jessica', 'William', 'Ashley']
        senders = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'Daniel Wilson', 'Emma Brown', 'Frank Miller']
        topics = ['marketing campaign', 'quarterly review', 'product launch', 'team building', 'budget planning', 
                 'client presentation', 'software update', 'training session', 'conference preparation', 'research project']
        
        ham_emails = []
        for _ in range(n_emails):
            template = random.choice(ham_templates)
            name = random.choice(names)
            sender = random.choice(senders)
            topic = random.choice(topics)
            
            email = template.format(name=name, sender=sender, topic=topic)
            ham_emails.append(email)
        
        return ham_emails
    
    @staticmethod
    def generate_spam_emails(n_emails: int) -> List[str]:
        """Generate spam email content"""
        spam_templates = [
            "CONGRATULATIONS!!! You have WON ${amount} in our EXCLUSIVE lottery! Click HERE to claim your PRIZE now! This offer is LIMITED TIME ONLY! Act fast before it expires! Free money guaranteed! No risk!",
            
            "URGENT: Your account needs immediate verification! Click this link NOW to secure your account: http://fake-bank-site.com. Failure to verify within 24 hours will result in account suspension. ACT NOW!",
            
            "Make ${amount} per day working from home! No experience needed! This amazing opportunity will change your life FOREVER! Click here to get started today! LIMITED spots available! GUARANTEED income!",
            
            "SPECIAL OFFER: Buy our miracle weight loss pills and lose {weight} pounds in just {days} days! GUARANTEED results or your money back! Order now and get FREE shipping! Limited time discount of {percent}% OFF!",
            
            "Dear valued customer, you have been selected for our EXCLUSIVE investment opportunity! Guaranteed returns of {percent}% per month! No risk involved! Send us your bank details to get started! ACT NOW before this offer expires!",
            
            "FREE MONEY ALERT! Claim your ${amount} government grant TODAY! No paperwork required! Everyone qualifies! Click here to apply now! This offer won't last long! Get your FREE money now!",
            
            "URGENT BUSINESS PROPOSAL: I am a prince from Nigeria and I need your help to transfer ${amount} million dollars. You will receive {percent}% commission for your assistance. Please reply with your bank account details. This is 100% LEGAL and SAFE!"
        ]
        
        spam_emails = []
        for _ in range(n_emails):
            template = random.choice(spam_templates)
            amount = random.randint(100, 50000)
            weight = random.randint(10, 50)
            days = random.randint(7, 30)
            percent = random.randint(10, 500)
            
            email = template.format(amount=amount, weight=weight, days=days, percent=percent)
            spam_emails.append(email)
        
        return spam_emails
    
    @staticmethod
    def generate_dataset(n_ham: int = 150, n_spam: int = 150) -> Tuple[List[str], List[int]]:
        """Generate complete email dataset"""
        random.seed(42)  # For reproducible results
        
        ham_emails = EmailDataGenerator.generate_ham_emails(n_ham)
        spam_emails = EmailDataGenerator.generate_spam_emails(n_spam)
        
        # Combine emails and labels
        emails = ham_emails + spam_emails
        labels = [0] * len(ham_emails) + [1] * len(spam_emails)  # 0 = ham, 1 = spam
        
        # Shuffle the dataset
        combined = list(zip(emails, labels))
        random.shuffle(combined)
        emails, labels = zip(*combined)
        
        return list(emails), list(labels)

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

def print_dataset_analysis(emails: List[str], labels: List[int]):
    """Print comprehensive dataset analysis"""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    n_total = len(emails)
    n_ham = sum(1 for label in labels if label == 0)
    n_spam = sum(1 for label in labels if label == 1)
    
    print(f"Total emails: {n_total}")
    print(f"Ham emails (legitimate): {n_ham} ({n_ham/n_total*100:.1f}%)")
    print(f"Spam emails: {n_spam} ({n_spam/n_total*100:.1f}%)")
    
    # Length analysis
    ham_lengths = [len(emails[i]) for i in range(len(emails)) if labels[i] == 0]
    spam_lengths = [len(emails[i]) for i in range(len(emails)) if labels[i] == 1]
    
    print(f"\nEmail Length Statistics:")
    print(f"Ham emails  - Average: {sum(ham_lengths)/len(ham_lengths):.1f} chars")
    print(f"Spam emails - Average: {sum(spam_lengths)/len(spam_lengths):.1f} chars")

def print_evaluation_results(y_true: List[int], y_pred: List[int], y_proba: List[float]):
    """Print comprehensive evaluation results"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    evaluator = ClassificationEvaluator()
    
    # Calculate all metrics
    accuracy = evaluator.accuracy(y_true, y_pred)
    precision = evaluator.precision(y_true, y_pred)
    recall = evaluator.recall(y_true, y_pred)
    f1 = evaluator.f1_score(y_true, y_pred)
    specificity = evaluator.specificity(y_true, y_pred)
    
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    # Confusion Matrix
    tp, fp, tn, fn = evaluator.confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                Ham  Spam")
    print(f"Actual Ham    {tn:4d} {fp:4d}")
    print(f"       Spam   {fn:4d} {tp:4d}")
    
    # Sample predictions
    print(f"\nSample Predictions (First 10 test cases):")
    print(f"{'True':<6} {'Pred':<6} {'Prob':<8} {'Result':<10}")
    print("-" * 35)
    for i in range(min(10, len(y_true))):
        true_label = "Spam" if y_true[i] == 1 else "Ham"
        pred_label = "Spam" if y_pred[i] == 1 else "Ham"
        result = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"{true_label:<6} {pred_label:<6} {y_proba[i]:<8.3f} {result:<10}")

def main():
    """Main function to demonstrate logistic regression for spam detection"""
    print_header()
    
    # Generate email dataset
    print("Generating email dataset for spam detection...")
    emails, labels = EmailDataGenerator.generate_dataset(n_ham=150, n_spam=150)
    
    # Analyze dataset
    print_dataset_analysis(emails, labels)
    
    # Text preprocessing and feature extraction
    print("\n" + "=" * 60)
    print("TEXT PREPROCESSING AND FEATURE EXTRACTION")
    print("=" * 60)
    
    print("\nExtracting TF-IDF features from email text...")
    vectorizer = TFIDFVectorizer(max_features=500, min_df=2)
    X_tfidf = vectorizer.fit_transform(emails)
    
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"Feature vector dimension: {len(X_tfidf[0]) if X_tfidf else 0}")
    
    # Additional features
    print("\nExtracting additional text features...")
    preprocessor = TextPreprocessor()
    additional_features = []
    
    for email in emails:
        features = preprocessor.extract_features(email)
        feature_vector = list(features.values())
        additional_features.append(feature_vector)
    
    # Combine TF-IDF with additional features
    X_combined = []
    for i in range(len(X_tfidf)):
        combined_vector = X_tfidf[i] + additional_features[i]
        X_combined.append(combined_vector)
    
    print(f"Combined feature dimension: {len(X_combined[0])} features")
    
    # Split dataset
    print("\nSplitting dataset into training (80%) and testing (20%)...")
    X_train, X_test, y_train, y_test = split_data(X_combined, labels, train_ratio=0.8)
    
    print(f"Training set: {len(X_train)} emails")
    print(f"Testing set: {len(X_test)} emails")
    
    # Train logistic regression model
    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    model = LogisticRegression()
    print(f"\nTraining with {len(X_combined[0])} features...")
    model.fit(X_train, y_train, learning_rate=0.1, iterations=1000)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    
    # Evaluate model
    print_evaluation_results(y_test, y_pred, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("SPAM DETECTION PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'test_data': (X_test, y_test),
        'predictions': (y_pred, y_pred_proba),
        'metrics': {
            'accuracy': ClassificationEvaluator.accuracy(y_test, y_pred),
            'f1_score': ClassificationEvaluator.f1_score(y_test, y_pred)
        }
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements logistic regression and text processing completely from scratch.
# 
# Key Dependencies:
# - math: For mathematical operations like log(), exp(), sqrt()
# - random: For data generation, shuffling, and parameter initialization
# - typing: For type hints (List, Tuple, Dict, Set)
# - re: For regular expression operations in text preprocessing
# 
# Educational Notes:
# 1. Logistic regression uses the sigmoid function to map any real number to (0,1)
# 2. Cross-entropy loss is the appropriate cost function for binary classification
# 3. TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numerical features
# 4. Text preprocessing is crucial for effective NLP applications
# 5. Feature engineering can significantly improve model performance
# 6. Confusion matrix provides detailed insight into classification performance
#
# This implementation demonstrates:
# - Binary classification with logistic regression
# - Comprehensive text preprocessing and feature extraction
# - TF-IDF vectorization for converting text to numerical features
# - Model evaluation with multiple classification metrics
# - Real-world application to spam detection problem
# - Gradient descent optimization for logistic regression parameters