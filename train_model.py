
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# 2. Separate input features and output labels
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model
joblib.dump(model, "crop_model.pkl")

# 8. Save the feature names for later use
joblib.dump(list(X.columns), "feature_names.pkl")

print("Model trained and saved successfully!")
