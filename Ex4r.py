from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd

# -------------------------------------
# STEP 1: INPUT DATASET
# -------------------------------------
documents = [
    # Autos (0)
    "I am planning to buy an electric car",
    "This vehicle has excellent fuel efficiency",
    "The car engine delivers strong performance",
    "Maintenance and servicing costs of cars are expensive",

    # Sports (1)
    "The football tournament was very thrilling",
    "The striker scored the winning goal",
    "Cricket players are practicing for the championship",
    "The game continued into extra time",

    # Technology (2)
    "The new smartphone features an advanced camera system",
    "Modern technology is evolving rapidly",
    "Artificial intelligence is transforming industries",
    "The latest mobile device uses innovative technology"
]

labels = [
    0, 0, 0, 0,  # Autos
    1, 1, 1, 1,  # Sports
    2, 2, 2, 2  # Technology
]

print("STEP 1: INPUT DATA")
for i, doc in enumerate(documents):
    print(f"{i + 1}. {doc}")
print("Labels:", labels)
print("-" * 60)

# Label mapping
label_names = {
    0: "Autos",
    1: "Sports",
    2: "Technology"
}

# -------------------------------------
# STEP 2: TRAIN-TEST SPLIT
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.4,
    random_state=42,
    stratify=labels
)

print("STEP 2: TRAINING DATA")
print(X_train)
print("Training Labels:", y_train)
print("\nTESTING DATA")
print(X_test)
print("Testing Labels:", y_test)
print("-" * 60)

# -------------------------------------
# STEP 3: TF-IDF VECTORIZATION
# -------------------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

tfidf_table = pd.DataFrame(
    X_train_vec.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("STEP 3: TF-IDF NUMERIC TABLE (TRAINING DATA)")
print(tfidf_table)
print("-" * 60)

# -------------------------------------
# STEP 4: MODELS
# -------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

# -------------------------------------
# STEP 5 & 6: TRAINING, PREDICTION & ACCURACY
# -------------------------------------
print("STEP 4–6: MODEL TRAINING, PREDICTION & ACCURACY\n")
results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    print(f"Algorithm: {name}")
    for text, pred in zip(X_test, predictions):
        print(f" Text: '{text}' → Predicted Class: {pred} ({label_names[pred]})")

    accuracy = accuracy_score(y_test, predictions)
    results.append([name, accuracy])
    print(f" Accuracy: {accuracy:.2f}")
    print("-" * 60)

# -------------------------------------
# STEP 7: FINAL ACCURACY COMPARISON
# -------------------------------------
accuracy_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy"])
print("\nFINAL ACCURACY COMPARISON")
print(accuracy_df)

print("\nClass Labels:")
print("0 → Autos")
print("1 → Sports")
print("2 → Technology")

