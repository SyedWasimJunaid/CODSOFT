import os
import pandas as pd
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("===== Movie Genre Classification =====\n")

# Get the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
train_path = os.path.join(BASE_DIR, "dataset", "train_data.txt")
test_path = os.path.join(BASE_DIR, "dataset", "test_data.txt")
solution_path = os.path.join(BASE_DIR, "dataset", "test_data_solution.txt")

# Check if files exist
if not os.path.exists(train_path):
    print("❌ train_data.txt not found!")
    print(train_path)
    exit()

if not os.path.exists(test_path):
    print("❌ test_data.txt not found!")
    print(test_path)
    exit()

if not os.path.exists(solution_path):
    print("❌ test_data_solution.txt not found!")
    print(solution_path)
    exit()

print("✅ All dataset files found.\n")

# Column names
train_columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]
test_columns = ["ID", "TITLE", "DESCRIPTION"]

# Read datasets
train_df = pd.read_csv(
    train_path,
    sep=" ::: ",
    engine="python",
    names=train_columns
)

test_df = pd.read_csv(
    test_path,
    sep=" ::: ",
    engine="python",
    names=test_columns
)

solution_df = pd.read_csv(
    solution_path,
    sep=" ::: ",
    engine="python",
    names=train_columns
)

# Display information
print("Training Dataset Shape :", train_df.shape)
print("Testing Dataset Shape  :", test_df.shape)
print("Solution Dataset Shape :", solution_df.shape)

print("\nFirst 5 rows of Training Dataset:\n")
print(train_df.head())

print("\nDataset loaded successfully!")

# -----------------------------
# Prepare Features and Labels
# -----------------------------

X_train = train_df["DESCRIPTION"]
y_train = train_df["GENRE"]

X_test = solution_df["DESCRIPTION"]
y_test = solution_df["GENRE"]

print("\nTraining samples :", len(X_train))
print("Testing samples  :", len(X_test))

# -----------------------------
# Naive Bayes Model
# -----------------------------

print("\nTraining Multinomial Naive Bayes Model...")

nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print(f"\nNaive Bayes Accuracy: {nb_accuracy:.4f}")

# -----------------------------
# Logistic Regression Model
# -----------------------------

print("\nTraining Logistic Regression Model...")

lr_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"\nLogistic Regression Accuracy: {lr_accuracy:.4f}")

# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, lr_predictions))

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, lr_predictions)

plt.figure(figsize=(15, 12))
sns.heatmap(cm, cmap="Blues")

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

confusion_matrix_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.show()

# -----------------------------
# Save Best Model
# -----------------------------

model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
joblib.dump(lr_model, model_path)

print("\nBest model saved successfully!")

# -----------------------------
# Predict Genre
# -----------------------------

while True:
    plot = input("\nEnter a movie plot (or type 'exit'): ")

    if plot.lower() == "exit":
        break

    prediction = lr_model.predict([plot])

    print("Predicted Genre:", prediction[0])