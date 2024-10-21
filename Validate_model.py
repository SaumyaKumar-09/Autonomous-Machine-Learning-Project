import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the serialized model
model = joblib.load(r'E:\Extion Infotech Internship\Task 1\best_model.joblib')

# Load the dataset for testing
test_data = pd.read_csv(r'E:\Extion Infotech Internship\Task 1\iris_test.csv')
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Make predictions using the loaded model
predictions = model.predict(X_test)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Validation Accuracy: {accuracy}')
