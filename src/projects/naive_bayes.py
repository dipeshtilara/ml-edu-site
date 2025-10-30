# Naive Bayes: Sentiment Analysis
# Text classification using Naive Bayes algorithm

import math
import random
from typing import List, Dict, Tuple
from collections import defaultdict

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for text sentiment analysis.
    Implements multinomial Naive Bayes with Laplace smoothing.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes classifier.
        
        Args:
            alpha: Laplace smoothing parameter
        """
        self.alpha = alpha
        self.class_priors = {}
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.classes = []
        
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization of text.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        # Convert to lowercase and split
        text = text.lower()
        # Remove punctuation
        for char in '.,!?;:\'"':
            text = text.replace(char, '')
        return text.split()
    
    def train(self, X_train: List[str], y_train: List[str]):
        """
        Train the Naive Bayes classifier.
        
        Args:
            X_train: List of text documents
            y_train: List of class labels
        """
        n_samples = len(X_train)
        self.classes = list(set(y_train))
        
        # Count class occurrences
        class_counts = defaultdict(int)
        for label in y_train:
            class_counts[label] += 1
        
        # Calculate class priors P(class)
        for label in self.classes:
            self.class_priors[label] = class_counts[label] / n_samples
        
        # Count word occurrences per class
        word_counts = {label: defaultdict(int) for label in self.classes}
        total_words = {label: 0 for label in self.classes}
        
        for text, label in zip(X_train, y_train):
            tokens = self.tokenize(text)
            for word in tokens:
                self.vocabulary.add(word)
                word_counts[label][word] += 1
                total_words[label] += 1
        
        # Calculate word probabilities P(word|class) with Laplace smoothing
        vocab_size = len(self.vocabulary)
        for label in self.classes:
            for word in self.vocabulary:
                count = word_counts[label][word]
                # Laplace smoothing
                prob = (count + self.alpha) / (total_words[label] + self.alpha * vocab_size)
                self.word_probs[label][word] = prob
    
    def predict(self, text: str) -> str:
        """
        Predict the class label for given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Predicted class label
        """
        tokens = self.tokenize(text)
        
        # Calculate log probabilities for each class
        class_scores = {}
        for label in self.classes:
            # Start with log of class prior
            score = math.log(self.class_priors[label])
            
            # Add log probabilities of words
            for word in tokens:
                if word in self.vocabulary:
                    score += math.log(self.word_probs[label][word])
            
            class_scores[label] = score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Predict class probabilities for given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary of class probabilities
        """
        tokens = self.tokenize(text)
        
        # Calculate log probabilities
        log_probs = {}
        for label in self.classes:
            log_prob = math.log(self.class_priors[label])
            for word in tokens:
                if word in self.vocabulary:
                    log_prob += math.log(self.word_probs[label][word])
            log_probs[label] = log_prob
        
        # Convert to probabilities using exp
        max_log_prob = max(log_probs.values())
        probs = {}
        for label in self.classes:
            probs[label] = math.exp(log_probs[label] - max_log_prob)
        
        # Normalize
        total = sum(probs.values())
        for label in probs:
            probs[label] /= total
        
        return probs
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """
        Evaluate classifier on test data.
        
        Args:
            X_test: Test text documents
            y_test: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = [self.predict(text) for text in X_test]
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
        accuracy = correct / len(y_test) * 100
        
        # Confusion matrix
        confusion = {}
        for true_label in self.classes:
            confusion[true_label] = {pred_label: 0 for pred_label in self.classes}
        
        for pred, true in zip(predictions, y_test):
            confusion[true][pred] += 1
        
        # Per-class metrics
        metrics = {}
        for label in self.classes:
            tp = confusion[label][label]
            fp = sum(confusion[other][label] for other in self.classes if other != label)
            fn = sum(confusion[label][other] for other in self.classes if other != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(y_test),
            'confusion_matrix': confusion,
            'per_class_metrics': metrics
        }
    
    def get_top_words(self, label: str, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N most important words for a class.
        
        Args:
            label: Class label
            n: Number of top words to return
            
        Returns:
            List of (word, probability) tuples
        """
        word_prob_list = [(word, prob) for word, prob in self.word_probs[label].items()]
        word_prob_list.sort(key=lambda x: x[1], reverse=True)
        return word_prob_list[:n]


def generate_sentiment_dataset(n_samples: int = 1000) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Generate synthetic sentiment analysis dataset.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    positive_words = [
        'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'love', 'perfect',
        'outstanding', 'brilliant', 'awesome', 'superb', 'incredible', 'best',
        'happy', 'delighted', 'satisfied', 'recommend', 'impressed', 'enjoyed'
    ]
    
    negative_words = [
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disappointing',
        'poor', 'useless', 'waste', 'failed', 'broken', 'defective',
        'angry', 'frustrated', 'disappointed', 'avoid', 'regret', 'unhappy'
    ]
    
    neutral_words = [
        'okay', 'average', 'normal', 'fine', 'acceptable', 'decent', 'alright',
        'product', 'service', 'item', 'purchase', 'received', 'ordered'
    ]
    
    common_words = [
        'the', 'this', 'is', 'was', 'it', 'very', 'really', 'quite', 'so',
        'i', 'my', 'me', 'have', 'had', 'been', 'would', 'could'
    ]
    
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        
        # Generate review based on sentiment
        if sentiment == 'positive':
            words = random.sample(positive_words, k=random.randint(3, 6))
            words += random.sample(common_words, k=random.randint(4, 8))
        elif sentiment == 'negative':
            words = random.sample(negative_words, k=random.randint(3, 6))
            words += random.sample(common_words, k=random.randint(4, 8))
        else:
            words = random.sample(neutral_words, k=random.randint(3, 5))
            words += random.sample(common_words, k=random.randint(4, 8))
            # Add small amount of positive or negative words
            if random.random() > 0.5:
                words += random.sample(positive_words, k=1)
            else:
                words += random.sample(negative_words, k=1)
        
        random.shuffle(words)
        review = ' '.join(words)
        
        X_data.append(review)
        y_data.append(sentiment)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * n_samples)
    X_train = X_data[:split_idx]
    y_train = y_data[:split_idx]
    X_test = X_data[split_idx:]
    y_test = y_data[split_idx:]
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    print("Naive Bayes: Sentiment Analysis")
    print("="*60)
    
    # Generate dataset
    print("\nGenerating sentiment analysis dataset...")
    X_train, y_train, X_test, y_test = generate_sentiment_dataset(1000)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier
    print("\nTraining Naive Bayes classifier...")
    nb = NaiveBayesClassifier(alpha=1.0)
    nb.train(X_train, y_train)
    print(f"Vocabulary size: {len(nb.vocabulary)}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = nb.evaluate(X_test, y_test)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    
    # Show top words per class
    for label in nb.classes:
        print(f"\nTop words for '{label}':")
        top_words = nb.get_top_words(label, n=10)
        for word, prob in top_words:
            print(f"  {word}: {prob:.4f}")
    
    print("\n" + "="*60)
    print("Sentiment Analysis Complete!")
