import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
print("Classes:", data.target_names)
class_counts = y.value_counts().sort_index()
total = len(y)
prob_malignant = class_counts[0] / total
prob_benign = class_counts[1] / total
print("\nProbability of Malignant:", prob_malignant)
print("Probability of Benign:", prob_benign)
if prob_benign > prob_malignant:
    prediction = "Benign"
else:
    prediction = "Malignant"
print("\nPredicted class for a new instance:", prediction)
