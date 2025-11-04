"""
Association Rules: Market Basket Analysis
Comprehensive implementation of Apriori algorithm for market basket analysis
CBSE Class 12 AI Project
"""

import json
from typing import List, Set, Tuple, Dict, Any
from itertools import combinations

class AprioriAlgorithm:
    """
    Apriori Algorithm for Association Rule Mining
    """
    
    def __init__(self, min_support: float = 0.2, min_confidence: float = 0.6):
        """
        Initialize Apriori algorithm
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        
    def get_support(self, itemset: frozenset, transactions: List[Set[str]]) -> float:
        """Calculate support for an itemset"""
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / len(transactions)
    
    def get_frequent_1_itemsets(self, transactions: List[Set[str]]) -> Dict[frozenset, float]:
        """Get frequent 1-itemsets"""
        item_counts = {}
        
        # Count occurrences
        for transaction in transactions:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Filter by minimum support
        n_transactions = len(transactions)
        frequent = {}
        
        for item, count in item_counts.items():
            support = count / n_transactions
            if support >= self.min_support:
                frequent[frozenset([item])] = support
        
        return frequent
    
    def generate_candidates(self, prev_frequent: List[frozenset], k: int) -> List[frozenset]:
        """Generate candidate itemsets of size k"""
        candidates = []
        n = len(prev_frequent)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Join two itemsets if they differ by only one item
                union = prev_frequent[i] | prev_frequent[j]
                if len(union) == k:
                    candidates.append(union)
        
        return list(set(candidates))
    
    def prune_candidates(self, candidates: List[frozenset], 
                        prev_frequent: Set[frozenset], k: int) -> List[frozenset]:
        """Prune candidates using Apriori property"""
        pruned = []
        
        for candidate in candidates:
            # Check if all (k-1) subsets are frequent
            subsets = [frozenset(s) for s in combinations(candidate, k - 1)]
            if all(subset in prev_frequent for subset in subsets):
                pruned.append(candidate)
        
        return pruned
    
    def fit(self, transactions: List[Set[str]]) -> Dict[str, Any]:
        """
        Run Apriori algorithm
        
        Args:
            transactions: List of transactions (each transaction is a set of items)
        
        Returns:
            Dictionary containing frequent itemsets
        """
        self.frequent_itemsets = {}
        
        # Get frequent 1-itemsets
        frequent_k = self.get_frequent_1_itemsets(transactions)
        self.frequent_itemsets[1] = frequent_k
        
        k = 2
        while True:
            # Generate candidates
            prev_itemsets = list(self.frequent_itemsets[k - 1].keys())
            candidates = self.generate_candidates(prev_itemsets, k)
            
            if not candidates:
                break
            
            # Prune candidates
            candidates = self.prune_candidates(
                candidates, 
                set(self.frequent_itemsets[k - 1].keys()), 
                k
            )
            
            if not candidates:
                break
            
            # Calculate support for candidates
            frequent_k = {}
            for candidate in candidates:
                support = self.get_support(candidate, transactions)
                if support >= self.min_support:
                    frequent_k[candidate] = support
            
            if not frequent_k:
                break
            
            self.frequent_itemsets[k] = frequent_k
            k += 1
        
        return self.frequent_itemsets
    
    def generate_rules(self) -> List[Dict[str, Any]]:
        """
        Generate association rules from frequent itemsets
        
        Returns:
            List of association rules with confidence and lift
        """
        self.rules = []
        
        # Generate rules from itemsets of size >= 2
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue
            
            for itemset, support in self.frequent_itemsets[k].items():
                # Generate all non-empty subsets as antecedents
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence
                        antecedent_support = None
                        for prev_k in range(1, k):
                            if prev_k in self.frequent_itemsets:
                                if antecedent in self.frequent_itemsets[prev_k]:
                                    antecedent_support = self.frequent_itemsets[prev_k][antecedent]
                                    break
                        
                        if antecedent_support is None or antecedent_support == 0:
                            continue
                        
                        confidence = support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            consequent_support = None
                            for prev_k in range(1, k):
                                if prev_k in self.frequent_itemsets:
                                    if consequent in self.frequent_itemsets[prev_k]:
                                        consequent_support = self.frequent_itemsets[prev_k][consequent]
                                        break
                            
                            if consequent_support is None or consequent_support == 0:
                                continue
                            
                            lift = confidence / consequent_support
                            
                            rule = {
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence,
                                'lift': lift
                            }
                            self.rules.append(rule)
        
        # Sort by confidence
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
        return self.rules


def generate_market_data():
    """Generate synthetic market basket data"""
    import random
    random.seed(42)
    
    items = {
        'Milk', 'Bread', 'Butter', 'Eggs', 'Cheese',
        'Coffee', 'Tea', 'Sugar', 'Flour', 'Rice',
        'Pasta', 'Sauce', 'Chicken', 'Fish', 'Beef',
        'Apple', 'Banana', 'Orange', 'Potato', 'Onion'
    }
    
    # Common item combinations (patterns)
    patterns = [
        ['Milk', 'Bread', 'Butter'],
        ['Coffee', 'Sugar'],
        ['Pasta', 'Sauce'],
        ['Eggs', 'Bread', 'Milk'],
        ['Chicken', 'Rice'],
        ['Apple', 'Banana', 'Orange'],
        ['Potato', 'Onion'],
        ['Tea', 'Sugar'],
        ['Bread', 'Butter', 'Cheese'],
        ['Fish', 'Rice']
    ]
    
    transactions = []
    
    # Generate 100 transactions
    for i in range(100):
        transaction = set()
        
        # Add a pattern with high probability
        if random.random() < 0.7:
            pattern = random.choice(patterns)
            transaction.update(pattern)
        
        # Add random items
        n_random = random.randint(1, 4)
        random_items = random.sample(list(items), n_random)
        transaction.update(random_items)
        
        transactions.append(transaction)
    
    return transactions, items


def main():
    """Main execution function"""
    print("=" * 70)
    print("Association Rules: Market Basket Analysis")
    print("=" * 70)
    print()
    
    # Generate data
    print("Step 1: Loading Transaction Data")
    print("-" * 70)
    transactions, items = generate_market_data()
    print(f"Loaded {len(transactions)} transactions")
    print(f"Total unique items: {len(items)}")
    print()
    
    # Display sample transactions
    print("Sample Transactions:")
    for i in range(5):
        print(f"  Transaction {i+1}: {', '.join(sorted(list(transactions[i])))[:50]}...")
    print()
    
    # Run Apriori algorithm
    print("Step 2: Running Apriori Algorithm")
    print("-" * 70)
    apriori = AprioriAlgorithm(min_support=0.15, min_confidence=0.5)
    frequent_itemsets = apriori.fit(transactions)
    
    print(f"Minimum support: {apriori.min_support*100:.1f}%")
    print(f"Minimum confidence: {apriori.min_confidence*100:.1f}%")
    print()
    
    # Display frequent itemsets
    print("Step 3: Frequent Itemsets")
    print("-" * 70)
    
    for k in sorted(frequent_itemsets.keys()):
        print(f"\n{k}-Itemsets (Count: {len(frequent_itemsets[k])})")
        
        # Show top 5 itemsets
        itemsets_sorted = sorted(
            frequent_itemsets[k].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for itemset, support in itemsets_sorted[:5]:
            items_str = ', '.join(sorted(list(itemset)))
            print(f"  {{{items_str}}} - Support: {support:.3f}")
        
        if len(itemsets_sorted) > 5:
            print(f"  ... and {len(itemsets_sorted) - 5} more")
    print()
    
    # Generate association rules
    print("Step 4: Generating Association Rules")
    print("-" * 70)
    rules = apriori.generate_rules()
    print(f"Total rules generated: {len(rules)}")
    print()
    
    # Display top rules
    print("Top 10 Association Rules:")
    print()
    
    for i, rule in enumerate(rules[:10], 1):
        antecedent = ', '.join(sorted(list(rule['antecedent'])))
        consequent = ', '.join(sorted(list(rule['consequent'])))
        
        print(f"{i}. {{{antecedent}}} => {{{consequent}}}")
        print(f"   Support: {rule['support']:.3f}, ", end="")
        print(f"   Confidence: {rule['confidence']:.3f}, ", end="")
        print(f"   Lift: {rule['lift']:.3f}")
        print()
    
    # Metrics explanation
    print("\nMetrics Explanation:")
    print("-" * 70)
    print("Support: How frequently the itemset appears in transactions")
    print("Confidence: How often the rule is true (P(Consequent|Antecedent))")
    print("Lift: How much more likely items are bought together vs. independently")
    print("  - Lift > 1: Positive correlation")
    print("  - Lift = 1: No correlation")
    print("  - Lift < 1: Negative correlation")
    print()
    
    # Business insights
    print("\n" + "=" * 70)
    print("Business Insights and Recommendations")
    print("=" * 70)
    print(f"✓ Discovered {len(rules)} strong association rules")
    print(f"✓ Identified frequent buying patterns")
    print()
    print("Applications:")
    print("• Product placement: Place associated items near each other")
    print("• Cross-selling: Recommend complementary products")
    print("• Bundle offers: Create product bundles based on rules")
    print("• Inventory management: Stock related items together")
    print("• Promotional campaigns: Target customers with relevant offers")
    print()

if __name__ == "__main__":
    main()
