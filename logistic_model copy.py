import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# ----------------------------------------
# 1. Load the dataset
# ----------------------------------------
data = pd.read_csv("balanced_logistic_dataset.csv")

# Features and target
X = data.drop("LoanDefault", axis=1)
y = data["LoanDefault"]

# ----------------------------------------
# 2. Train / Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------------------
# 3. Build Logistic Regression Model
# ----------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------------
# 4. Save Model as Pickle
# ----------------------------------------
pickle_filename = "loan_default_model.pkl"
with open(pickle_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as: {pickle_filename}")

# ----------------------------------------
# 5. Load the Pickle File and Test
# ----------------------------------------
with open(pickle_filename, "rb") as file:
    loaded_model = pickle.load(file)

# Make sample predictions
predictions = loaded_model.predict(X_test[:5])
print("\nSample Predictions:", predictions)

# Actual values
print("Actual Values:", list(y_test[:5]))
