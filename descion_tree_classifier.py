import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Load the dataset
file_path = "C:\\Users\\srikr\\Desktop\\COLLEGE\\Extra\\bank_1.csv"  # Update with the actual path to your file
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Drop the 'duration' column as it is not known beforehand
df = df.drop('duration', axis=1)

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop('deposit_yes', axis=1)  # Features
y = df['deposit_yes']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Display the decision tree
tree.plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True, rounded=True)
