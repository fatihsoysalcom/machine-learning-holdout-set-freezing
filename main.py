import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Generate synthetic data for a binary classification task.
# We'll create a dataset with 1000 samples, 20 features, and 2 classes.
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

print("Original dataset shape:", X.shape, y.shape)

# 2. Split the data into an initial training+validation set and a separate, frozen holdout (test) set.
# The holdout set is crucial; it will be used ONLY ONCE at the very end to evaluate the final model.
# It simulates truly unseen real-world data, preventing data leakage.
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining + Validation set shape: {X_train_val.shape}, {y_train_val.shape}")
print(f"Frozen Holdout (Test) set shape: {X_holdout.shape}, {y_holdout.shape}")

# 3. Further split the training+validation set into actual training and validation sets.
# The training set is used to train the model.
# The validation set is used for hyperparameter tuning and model selection during development.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
) # 0.25 of 0.8 (train_val) results in 0.2 of the original data, creating a 60-20-20 split.

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")

# 4. Train a simple model (Logistic Regression) using only the training set.
# In a real scenario, you might iterate here, training multiple models or tuning hyperparameters.
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained using the training set.")

# 5. Evaluate the model on the validation set.
# This helps in selecting the best model or hyperparameters without touching the holdout set.
val_predictions = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Model accuracy on Validation set: {val_accuracy:.4f}")

# --- IMPORTANT: The holdout set remains untouched until this point ---

# 6. Finally, evaluate the chosen model ONLY ONCE on the frozen holdout set.
# This provides an unbiased estimate of the model's performance on truly unseen data.
# This score is what you would report as the "Coding Agent Score" or final performance metric.
holdout_predictions = model.predict(X_holdout)
holdout_accuracy = accuracy_score(y_holdout, holdout_predictions)
print(f"\n--- FINAL EVALUATION ---")
print(f"Model accuracy on FROZEN HOLDOUT (TEST) set: {holdout_accuracy:.4f}")
print("\nThis holdout score is the unbiased estimate of the model's real-world performance.")
print("It was not used during training or hyperparameter tuning, preventing data leakage.")
