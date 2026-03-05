import math

documents = [
    "data mining techniques",
    "data mining process",
    "data analysis methods"
]

N = len(documents)

print("\nDOCUMENT COLLECTION")
print("-" * 80)
for i, doc in enumerate(documents):
    print(f"D{i+1}: {doc}")

# -----------------------------
# QUERY
# -----------------------------
query = input("\nEnter the query: ").lower().split()

# -----------------------------
# WORD FREQUENCY
# -----------------------------
word_freq = []
vocab = set()

for doc in documents:
    freq = {}
    for word in doc.split():
        freq[word] = freq.get(word, 0) + 1
        vocab.add(word)
    word_freq.append(freq)

vocab = sorted(vocab)

print("\nWORD FREQUENCY TABLE - no.of times the word occurs in each document")
print("-" * 80)
print("Word".ljust(15), end="")
for i in range(N):
    print(f"D{i+1}".ljust(8), end="")
print()

for w in vocab:
    print(w.ljust(15), end="")
    for i in range(N):
        print(str(word_freq[i].get(w, 0)).ljust(8), end="")
    print()

# -----------------------------
# TERM FREQUENCY (LOG SCALED)
# -----------------------------
tf = []

for freq in word_freq:
    tf_doc = {}
    for w in freq:
        tf_doc[w] = round(1 + math.log10(freq[w]), 2)
    tf.append(tf_doc)

print("\nTERM FREQUENCY (LOG SCALED)")
print("-" * 80)
print("Word".ljust(15), end="")
for i in range(N):
    print(f"D{i+1}".ljust(8), end="")
print()

for w in vocab:
    print(w.ljust(15), end="")
    for i in range(N):
        print(str(tf[i].get(w, 0)).ljust(8), end="")
    print()

# -----------------------------
# DF and IDF
# -----------------------------
df = {}
idf = {}

for w in vocab:
    df[w] = sum(1 for doc in documents if w in doc.split())
    idf[w] = round(math.log10(N / df[w]), 2)

print("\nIDF TABLE --> log(N/DF)")
print("-" * 80)
print("Word".ljust(15), "N".ljust(6), "DF".ljust(6), "log(N/DF)")

for w in vocab:
    print(w.ljust(15), str(N).ljust(6), str(df[w]).ljust(6), idf[w])

# -----------------------------
# TF-IDF
# -----------------------------
tfidf = []

for i in range(N):
    tfidf_doc = {}
    for w in vocab:
        tfidf_doc[w] = round(tf[i].get(w, 0) * idf[w], 2)
    tfidf.append(tfidf_doc)

print("\nTF-IDF = TF × IDF")
print("-" * 80)
print("Word".ljust(15), end="")
for i in range(N):
    print(f"D{i+1}".ljust(8), end="")
print()

for w in vocab:
    print(w.ljust(15), end="")
    for i in range(N):
        print(str(tfidf[i][w]).ljust(8), end="")
    print()

# -----------------------------
# RELEVANCE CALCULATION
# -----------------------------
print("\nRELEVANCE CALCULATION = SUMMATION of QUERY TF-IDF")
print("-" * 80)
print("Doc".ljust(6), end="")
for w in query:
    print(w.ljust(12), end="")
print("Score")

scores = []

for i in range(N):
    score = 0
    print(f"D{i+1}".ljust(6), end="")
    for w in query:
        val = tfidf[i].get(w, 0)
        score += val
        print(str(val).ljust(12), end="")
    scores.append(score)
    print(score)

# -----------------------------
# FINAL VERDICT
# -----------------------------
print("\nFINAL VERDICT")
print("-" * 80)

max_score = max(scores)

if max_score == 0 and all(w in vocab for w in query):
    print("Query exists in ALL documents.")
elif max_score == 0:
    print("Query terms do NOT exist in the document collection.")
else:
    print("Most Relevant Document(s):\n")
    for i in range(N):
        if scores[i] == max_score:
            print(f"D{i+1}: {documents[i]}")
