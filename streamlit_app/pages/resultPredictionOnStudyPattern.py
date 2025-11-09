

# Import libraries
import pandas as pd   # to manage tabular data
from sklearn.model_selection import train_test_split # splits dataset into training and testing parts
from sklearn.linear_model import LogisticRegression # model used for classification
from sklearn.metrics import confusion_matrix, classification_report # for evaluating model performance
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for visualization

# Sample dataset (replace with your actual df)
data = {
    'Study Hours': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'Attendance': [80, 85, 78, 90, 88, 92, 95, 98],
    'Interest': [3, 4, 2, 5, 4, 5, 5, 5],
    'Test Scores': [40, 50, 45, 60, 65, 70, 75, 80]
}
df = pd.DataFrame(data)

# Step 1: Create classification labels (Pass = 1, Fail = 0)
df['Result'] = df['Test Scores'].apply(lambda x: 1 if x >= 50 else 0)

# Step 2: Features & Target
X = df[['Study Hours', 'Attendance', 'Interest']]
y = df['Result']   # output/Target

# Step 3: Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 4: Train Logistic Regression classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 5: Predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

'''
|                     | Predicted Fail (0)  | Predicted Pass (1)  |
| ------------------- | ------------------- | ------------------- |
| **Actual Fail (0)** | True Negative (TN)  | False Positive (FP) |
| **Actual Pass (1)** | False Negative (FN) | True Positive (TP)  |
'''

# Step 7: Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
