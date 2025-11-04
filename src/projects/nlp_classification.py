"""
NLP: Multi-class Text Classification System
Comprehensive implementation of NLP pipeline for text classification
CBSE Class 12 AI Project
"""

import json
import re
import math
from typing import List, Tuple, Dict, Any, Set
from collections import Counter

class TextPreprocessor:
    """
    Text preprocessing and tokenization
    """
    
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        ])
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens"""
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def preprocess(self, text: str) -> List[str]:
        """Complete preprocessing pipeline"""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer
    """
    
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.n_documents = 0
    
    def fit(self, documents: List[List[str]]):
        """Fit vectorizer on documents"""
        self.n_documents = len(documents)
        
        # Build vocabulary
        all_tokens = set()
        for doc in documents:
            all_tokens.update(doc)
        
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Calculate IDF
        document_frequency = Counter()
        for doc in documents:
            unique_tokens = set(doc)
            for token in unique_tokens:
                document_frequency[token] += 1
        
        for token, df in document_frequency.items():
            self.idf[token] = math.log(self.n_documents / df)
    
    def transform(self, document: List[str]) -> List[float]:
        """Transform document to TF-IDF vector"""
        # Calculate term frequency
        tf = Counter(document)
        total_terms = len(document)
        
        # Create TF-IDF vector
        vector = [0.0] * len(self.vocabulary)
        
        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_value = count / total_terms
                idf_value = self.idf.get(token, 0)
                vector[idx] = tf_value * idf_value
        
        # Normalize vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector
    
    def fit_transform(self, documents: List[List[str]]) -> List[List[float]]:
        """Fit and transform documents"""
        self.fit(documents)
        return [self.transform(doc) for doc in documents]


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes Classifier for text classification
    """
    
    def __init__(self, alpha: float = 1.0):
        """Initialize classifier with Laplace smoothing"""
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
        self.n_features = 0
    
    def fit(self, X: List[List[float]], y: List[str]):
        """Train classifier"""
        self.classes = list(set(y))
        self.n_features = len(X[0]) if X else 0
        
        # Calculate class priors
        class_counts = Counter(y)
        total_samples = len(y)
        
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / total_samples
        
        # Calculate feature probabilities for each class
        for class_label in self.classes:
            # Get all samples of this class
            class_samples = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            # Sum features across all samples
            feature_sums = [0.0] * self.n_features
            for sample in class_samples:
                for j, value in enumerate(sample):
                    feature_sums[j] += value
            
            # Calculate probabilities with Laplace smoothing
            total_features = sum(feature_sums) + self.alpha * self.n_features
            
            self.feature_probs[class_label] = [
                (feature_sum + self.alpha) / total_features
                for feature_sum in feature_sums
            ]
    
    def predict_proba(self, x: List[float]) -> Dict[str, float]:
        """Predict class probabilities for a sample"""
        log_probs = {}
        
        for class_label in self.classes:
            # Start with log prior
            log_prob = math.log(self.class_priors[class_label])
            
            # Add log likelihood
            for j, value in enumerate(x):
                if value > 0:
                    log_prob += value * math.log(self.feature_probs[class_label][j])
            
            log_probs[class_label] = log_prob
        
        # Convert to probabilities (softmax)
        max_log_prob = max(log_probs.values())
        exp_probs = {k: math.exp(v - max_log_prob) for k, v in log_probs.items()}
        total = sum(exp_probs.values())
        
        return {k: v / total for k, v in exp_probs.items()}
    
    def predict(self, x: List[float]) -> str:
        """Predict class label for a sample"""
        probs = self.predict_proba(x)
        return max(probs, key=probs.get)


def generate_text_data():
    """Generate synthetic text classification data"""
    # Technology news
    tech_texts = [
        "New smartphone features advanced AI processor and 5G connectivity",
        "Cloud computing platform launches with machine learning capabilities",
        "Software update brings improved security and performance enhancements",
        "Tech company announces breakthrough in quantum computing research",
        "Latest laptop features high-resolution display and powerful graphics card",
        "Artificial intelligence system achieves human-level performance in tests",
        "New programming language designed for data science applications",
        "Cybersecurity experts warn about emerging threats in digital landscape",
        "Virtual reality headset offers immersive gaming experience",
        "Database technology improves query performance by 10x"
    ]
    
    # Sports news
    sports_texts = [
        "Team wins championship after incredible comeback in final quarter",
        "Star player breaks record with outstanding performance in match",
        "Coach announces new strategy for upcoming tournament season",
        "Olympic athlete qualifies with record-breaking time in trials",
        "Football team secures victory with last-minute goal",
        "Basketball player scores 50 points in playoff game",
        "Tennis champion advances to finals after defeating rival",
        "Swimmer breaks world record in freestyle competition",
        "Cricket team chases target successfully in thrilling finish",
        "Athlete wins gold medal in marathon race"
    ]
    
    # Business news
    business_texts = [
        "Stock market reaches new high as investors show confidence",
        "Company reports strong quarterly earnings and revenue growth",
        "Merger deal between major corporations approved by regulators",
        "Startup raises millions in funding from venture capital firms",
        "Central bank announces interest rate decision affecting economy",
        "Retail sales increase as consumer spending shows strength",
        "Trade agreement signed between countries boosting exports",
        "Business leader appointed as CEO of multinational company",
        "Financial report shows profit increase in third quarter",
        "Economic indicators suggest growth in manufacturing sector"
    ]
    
    # Create dataset
    texts = []
    labels = []
    
    for text in tech_texts:
        texts.append(text)
        labels.append('Technology')
    
    for text in sports_texts:
        texts.append(text)
        labels.append('Sports')
    
    for text in business_texts:
        texts.append(text)
        labels.append('Business')
    
    return texts, labels


def main():
    """Main execution function"""
    print("=" * 70)
    print("NLP: Multi-class Text Classification System")
    print("=" * 70)
    print()
    
    # Generate data
    print("Step 1: Loading Text Dataset")
    print("-" * 70)
    texts, labels = generate_text_data()
    print(f"Total documents: {len(texts)}")
    print(f"Categories: {set(labels)}")
    print()
    
    print("Sample documents:")
    for i in range(3):
        print(f"  [{labels[i]}] {texts[i][:50]}...")
    print()
    
    # Preprocessing
    print("Step 2: Text Preprocessing")
    print("-" * 70)
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for text in texts]
    
    print("Original text:")
    print(f"  {texts[0]}")
    print("After preprocessing:")
    print(f"  {' '.join(processed_texts[0])}")
    print()
    
    avg_tokens = sum(len(doc) for doc in processed_texts) / len(processed_texts)
    print(f"Average tokens per document: {avg_tokens:.1f}")
    print()
    
    # Vectorization
    print("Step 3: TF-IDF Vectorization")
    print("-" * 70)
    vectorizer = TFIDFVectorizer()
    X = vectorizer.fit_transform(processed_texts)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"Vector dimensions: {len(X[0])}")
    print()
    
    # Show top TF-IDF terms for first document
    first_vector = X[0]
    top_indices = sorted(range(len(first_vector)), 
                        key=lambda i: first_vector[i], 
                        reverse=True)[:5]
    
    reverse_vocab = {idx: word for word, idx in vectorizer.vocabulary.items()}
    print("Top 5 TF-IDF terms in first document:")
    for idx in top_indices:
        if first_vector[idx] > 0:
            print(f"  {reverse_vocab[idx]}: {first_vector[idx]:.4f}")
    print()
    
    # Train classifier
    print("Step 4: Training Naive Bayes Classifier")
    print("-" * 70)
    
    # Split data (simple split for demonstration)
    train_size = int(len(X) * 0.7)
    X_train, y_train = X[:train_size], labels[:train_size]
    X_test, y_test = X[train_size:], labels[train_size:]
    
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.fit(X_train, y_train)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {classifier.classes}")
    print()
    
    # Evaluate
    print("Step 5: Model Evaluation")
    print("-" * 70)
    
    # Predictions
    predictions = [classifier.predict(x) for x in X_test]
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    print(f"Accuracy: {accuracy:.2%}")
    print()
    
    # Show predictions
    print("Sample Predictions:")
    for i in range(min(5, len(X_test))):
        idx = train_size + i
        probs = classifier.predict_proba(X_test[i])
        pred = predictions[i]
        true = y_test[i]
        
        status = "✓" if pred == true else "✗"
        print(f"  {status} Text: {texts[idx][:40]}...")
        print(f"    True: {true}, Predicted: {pred}")
        print(f"    Probabilities: ", end="")
        print(", ".join([f"{k}: {v:.2%}" for k, v in sorted(probs.items())]))
        print()
    
    # Per-class accuracy
    print("Per-class Performance:")
    for class_label in classifier.classes:
        class_indices = [i for i, label in enumerate(y_test) if label == class_label]
        if class_indices:
            class_correct = sum(1 for i in class_indices if predictions[i] == class_label)
            class_accuracy = class_correct / len(class_indices)
            print(f"  {class_label}: {class_accuracy:.2%} ({class_correct}/{len(class_indices)})")
    print()
    
    # Summary
    print("\n" + "=" * 70)
    print("NLP Classification Summary")
    print("=" * 70)
    print(f"✓ Processed {len(texts)} text documents")
    print(f"✓ Built vocabulary of {len(vectorizer.vocabulary)} terms")
    print(f"✓ Trained Naive Bayes classifier")
    print(f"✓ Achieved {accuracy:.1%} classification accuracy")
    print()
    print("Key Techniques:")
    print("• Text preprocessing: tokenization, stop word removal")
    print("• TF-IDF: captures term importance across documents")
    print("• Naive Bayes: probabilistic classification with independence assumption")
    print("• Multi-class classification: handles multiple categories")
    print()

if __name__ == "__main__":
    main()
