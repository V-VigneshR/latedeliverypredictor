import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Replace with your actual dataset loading code
import shap
import json

# Create model directory if it doesn't exist
model_directory = 'model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Load your dataset (replace with your actual dataset)
data = load_iris()  # Example dataset, replace with your dataset
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_classifier, os.path.join(model_directory, 'random_forest_classifier.pkl'))

print("Model trained and saved as 'model/random_forest_classifier.pkl'")

# Use SHAP to explain the model
explainer = shap.TreeExplainer(rf_classifier)

# Generate SHAP values for the test set (or for a specific input instance)
shap_values = explainer.shap_values(X_test)

# Save SHAP values as JSON to pass to the dashboard
shap_json = json.dumps(shap_values.tolist())
with open(os.path.join(model_directory, 'shap_values.json'), 'w') as f:
    f.write(shap_json)

# Save feature importance data
feature_importances = rf_classifier.feature_importances_.tolist()
importance_json = json.dumps(feature_importances)
with open(os.path.join(model_directory, 'feature_importances.json'), 'w') as f:
    f.write(importance_json)

print("SHAP values and feature importances saved as JSON files.")
