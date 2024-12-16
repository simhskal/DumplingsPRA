import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# 0:Cat, 1:Dog, 2:Squirrel
predictions = [0, 0, 1, 1, 0, 0, 2, 2, 1, 0]
ground_truth = [0, 1, 1, 1, 1, 0, 2, 2, 2, 2]

# Compute the confusion matrix
cm = confusion_matrix(ground_truth, predictions)

# Compute accuracy, precision, and recall
accuracy = accuracy_score(ground_truth, predictions)
report = classification_report(ground_truth, predictions, target_names=['Cat', 'Dog', 'Squirrel'])


# Visualize the confusion matrix with class names
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Cat', 'Dog', 'Squirrel'], yticklabels=['Cat', 'Dog', 'Squirrel'], ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()

# Output the classification report 
print("Accuracy:", accuracy)
print("\nClassification report:\n", report)