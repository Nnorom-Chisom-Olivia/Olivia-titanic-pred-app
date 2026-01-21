import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, 
recall_score, f1_score, confusion_matrix)
from sklearn.metrics import roc_curve, auc

# Load the dataset (either using the URL or local file path)
data = pd.read_csv("train.csv")

#preview data
pd.set_option("display.max_columns", None)
print(data.head())
print(data.info())

 # Data Preprocessing; Drop columns that are not useful for prediction
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'Embarked'], axis=1)

data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})


 # Handle missing values
 # Fill missing age values with median
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Fare'] = data['Fare'].fillna(data['Fare'].median())


#preview after preprocessing
print('\nAfter preprocessing:')
print(data.head())
print(data.info())

# Split the data into features (X) and target variable (y)
X = data.drop('Survived', axis=1)
y = data['Survived']
 
 #Step 5: Split the Dataset into Training and Testing Sets
 # Use an 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=42)

#Train the Logistic Regression Model
#Initialize the Logistic Regression Model
model = LogisticRegression(max_iter=1000)

#Fit the model on the training data
model.fit(X_train, y_train)
print('Training is complete')

#Step 7: Make Predictions
#Use the trained model to make predictions on the test set
y_pred = model.predict(X_test)
print('\nprediction:')
print(y_pred)

#Evaluate the Model's Performance using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#print evaluation results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

#Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Generate the Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# plotting the ROC curve to visualize the classification performance.
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, 
model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig("roc_curve.png", dpi=300)
plt.show()

import os
import joblib

# Create the folder if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Updated save paths for Scorac Part C
joblib.dump(X.columns.tolist(), "model/titanic_columns.joblib")
joblib.dump(model, "model/titanic_survival_model.pkl")


"""confusion matrics explanation:TP=55,TN=89, FP=16, FN=19.
The model achieved: Accuracy=79.3%, Precision=76.8%, Recall=72%, F1 score=74.1%, and AUC of 88%.
These results show that the model is well balanced, with good predictive power and strong ability to distinguish between the two classes. 
The high AUC score confirms that the model generizes well and is effective for Binary classification Tasks."""