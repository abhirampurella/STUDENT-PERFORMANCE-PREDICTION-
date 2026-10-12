import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =================================
# LOAD DATASET
# =================================

data = pd.read_csv("dataset/student_data.csv")

print("Dataset loaded successfully!")
print(data.head())


# =================================
# FEATURES AND TARGET
# =================================

X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score",
        "sleep_hours"
    ]
]

y = data["result"]


# Convert Pass/Fail to numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# =================================
# TRAIN / TEST SPLIT
# =================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =================================
# DECISION TREE
# =================================

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_prediction = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_prediction
)


# =================================
# RANDOM FOREST
# =================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_prediction = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)


# =================================
# LOGISTIC REGRESSION
# =================================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(X_train, y_train)

lr_prediction = lr_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_prediction
)


# =================================
# MODEL COMPARISON
# =================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    "Decision Tree Accuracy:",
    round(dt_accuracy, 2)
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy, 2)
)

print(
    "Logistic Regression Accuracy:",
    round(lr_accuracy, 2)
)


# Accuracy dictionary
accuracies = {
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy,
    "Logistic Regression": lr_accuracy
}


# Find best model name
best_model_name = max(
    accuracies,
    key=accuracies.get
)

print("\nBest Model:", best_model_name)

print(
    "Best Accuracy:",
    round(accuracies[best_model_name], 2)
)


# =================================
# EVALUATION
# =================================

print("\n==============================")
print("RANDOM FOREST EVALUATION")
print("==============================")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_prediction,
        labels=[0, 1],
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_prediction,
        labels=[0, 1]
    )
)


# =================================
# SELECT ACTUAL BEST MODEL
# =================================

trained_models = {
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Logistic Regression": lr_model
}

best_model = trained_models[best_model_name]


# =================================
# SAVE BEST MODEL + ENCODER
# =================================

joblib.dump(
    best_model,
    "student_model.pkl"
)

joblib.dump(
    encoder,
    "label_encoder.pkl"
)

print("\nBest model saved successfully!")
print("Saved model:", best_model_name)
