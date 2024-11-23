import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
# (ii) Perform preprocessing
breast_cancer = load_breast_cancer() # Load the dataset
data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
data['TARGET'] = breast_cancer.target
# Display basic information
print("First few rows of the dataset:",data.head())
print("\nSummary statistics:",data.describe())
print("\nInformation about the dataset:",data.info())
# Check for missing values
print("\nMissing values in each column:",data.isnull().sum())
# (iii) Split the dataset# Features and target variable
features = data.drop('TARGET', axis=1)
target = data['TARGET']
scaler = StandardScaler() # Standardize features
features_scaled = scaler.fit_transform(features)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
# (iv) Build and train the decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) # Predict on the test set
accuracy = accuracy_score(y_test, y_pred) # Find the accuracy
print(f"\nAccuracy of the Decision Tree model: {accuracy:.2f}")
# Display classification report
print("\nClassification Report:",classification_report(y_test,y_pred))
# (v) Data prediction #Predicting on a sample data point
sample_data = X_test[0].reshape(1, -1) # Take the first sample from the test set
sample_prediction = clf.predict(sample_data)
print(f"\nPrediction for sample data point: {sample_prediction[0]}")
# Optional: Visualize the decision tree
from sklearn import tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=breast_cancer.feature_names, class_names=breast_cancer.target_names,
filled=True)
plt.show()
