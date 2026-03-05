import math
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Documents
documents = [
    "Information retrieval systems help users find relevant documents from large datasets efficiently",
    "Techniques in information retrieval improve search accuracy for research papers and text data",
    "Machine learning methods are applied to data mining for extracting useful patterns from datasets"
]

# User query
query = input("Enter your search query: ").strip()
print("\nUser Query:", query)

# ---------------- Tokenization ----------------
def tokenize(text):
    words = text.lower().split()
    return [word for word in words if word not in stop_words]

tokenized_docs = [tokenize(doc) for doc in documents]
tokenized_query = tokenize(query)

print("\nTokenized Documents (After Stop Word Removal)")
for i, doc in enumerate(tokenized_docs, start=1):
    print(f"D{i}:", doc)

print("Query Tokens:", tokenized_query)

# ---------------- Vocabulary ----------------
vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
print("\nVocabulary:", vocab)

# ---------------- TF ----------------
def compute_tf(tokens):
    counts = Counter(tokens)
    total = len(tokens)
    return {word: counts[word]/total if total > 0 else 0 for word in vocab}

tf_docs = [compute_tf(doc) for doc in tokenized_docs]
tf_query = compute_tf(tokenized_query)

print("\nTF Table")
print(f"{'Term':<15}{'D1':<10}{'D2':<10}{'D3':<10}{'Query':<10}")
for word in vocab:
    print(f"{word:<15}{tf_docs[0][word]:<10.3f}{tf_docs[1][word]:<10.3f}{tf_docs[2][word]:<10.3f}{tf_query[word]:<10.3f}")

# ---------------- IDF ----------------
N = len(documents)
idf = {}

print("\nIDF Calculation")
for word in vocab:
    df = sum(1 for doc in tokenized_docs if word in doc)
    idf[word] = math.log(N/df) if df != 0 else 0
    print(f"{word:<15} DF={df}  IDF=log({N}/{df}) = {idf[word]:.3f}")

# ---------------- TF-IDF ----------------
tfidf_docs = [{word: tf_docs[i][word]*idf[word] for word in vocab} for i in range(N)]
tfidf_query = {word: tf_query[word]*idf[word] for word in vocab}

print("\nTF-IDF Table")
print(f"{'Term':<15}{'D1':<10}{'D2':<10}{'D3':<10}{'Query':<10}")
for word in vocab:
    print(f"{word:<15}{tfidf_docs[0][word]:<10.3f}{tfidf_docs[1][word]:<10.3f}{tfidf_docs[2][word]:<10.3f}{tfidf_query[word]:<10.3f}")

# ---------------- Cosine Similarity ----------------
def cosine_similarity(vec1, vec2):

    print("\nDot Product Calculation:")
    dot = 0
    for word in vocab:
        mul = vec1[word] * vec2[word]
        if mul != 0:
            print(f"{word}: {vec1[word]:.3f} × {vec2[word]:.3f} = {mul:.6f}")
        dot += mul

    mag1 = math.sqrt(sum(vec1[word]**2 for word in vocab))
    mag2 = math.sqrt(sum(vec2[word]**2 for word in vocab))

    print("\nDot Product =", dot)
    print("Magnitude Document =", mag1)
    print("Magnitude Query =", mag2)

    if mag1 and mag2:
        similarity = dot/(mag1*mag2)
        print(f"Similarity = {dot} / ({mag1} × {mag2}) = {similarity}")
        return similarity
    else:
        return 0

# ---------------- Similarity for Each Document ----------------
print("\nCosine Similarity Calculations")
similarities = []

for i in range(N):
    print(f"\n--- D{i+1} ---")
    sim = cosine_similarity(tfidf_docs[i], tfidf_query)
    similarities.append(sim)
    print(f"Similarity D{i+1}: {sim:.3f}")

# ---------------- Ranking ----------------
ranked_indices = sorted(range(N), key=lambda i: similarities[i], reverse=True)
best_idx = ranked_indices[0]

print("\nFinal Ranking")
print(f"{'Rank':<6}{'Document':<10}{'Similarity'}")
for rank, idx in enumerate(ranked_indices, start=1):
    print(f"{rank:<6}D{idx+1:<10}{similarities[idx]:.3f}")

# ---------------- Best Abstract ----------------
print("\nBest Abstract")
print(f"D{best_idx+1}: {documents[best_idx]}")
print(f"Similarity Score: {similarities[best_idx]:.3f}")
print(f"User Query: \"{query}\"")
