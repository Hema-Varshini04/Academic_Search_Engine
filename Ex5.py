from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Machine learning enhances recommendation systems",
    "Search systems rely on information retrieval techniques",
    "Document clustering groups similar texts together",
    "Deep neural networks are driving artificial intelligence",
    "Ranking algorithms determine search result relevance"
]

print("Documents:\n")
for i, doc in enumerate(documents, start=1):
    print(f"{i}. {doc}")

# -----------------------------
# QUERY
# -----------------------------
query = input("\nEnter your search query: ")

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_

# -----------------------------
# TF, IDF, TF-IDF FOR QUERY
# -----------------------------
print("\n--- TF, IDF, TF-IDF for Query ---")

query_tokens = query.lower().split()
print("{:<15} {:<5} {:<8} {:<8}".format("Term", "TF", "IDF", "TF-IDF"))

for term, idf_val in zip(feature_names, idf_values):
    tf = query_tokens.count(term)
    tfidf_val = tf * idf_val
    print("{:<15} {:<5} {:<8.3f} {:<8.3f}".format(term, tf, idf_val, tfidf_val))

# -----------------------------
# TF-IDF FOR EACH DOCUMENT
# -----------------------------
print("\n--- TF-IDF for Each Document ---")

for i, doc in enumerate(documents):
    doc_vector = doc_vectors[i].toarray()[0]
    print(f"\nDocument {i + 1}: {doc}")
    print("{:<15} {:<8}".format("Term", "TF-IDF"))
    for term, val in zip(feature_names, doc_vector):
        print("{:<15} {:<8.3f}".format(term, val))

# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------
combined_vectors = np.vstack([query_vector.toarray(), doc_vectors.toarray()])

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(combined_vectors)

query_cluster = labels[0]
print(f"\nQuery belongs to Cluster: {query_cluster}")

clusters = {}
for i, label in enumerate(labels[1:]):
    clusters.setdefault(label, []).append(documents[i])

print("\n--- Cluster Assignments ---")
for c in clusters:
    print(f"\nCluster {c}:")
    for doc in clusters[c]:
        print(" -", doc)

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
print("\n--- Cosine Similarity with Query ---")

for i, doc in enumerate(documents):
    doc_vector = doc_vectors[i].toarray()[0]
    cos_sim = cosine_similarity(query_vector, doc_vectors[i])[0][0]

    print(f"\nDocument {i + 1}: {doc}")
    print("{:<15} {:<12} {:<12} {:<12}".format(
        "Term", "TF-IDF_Doc", "TF-IDF_Q", "Product"
    ))

    for term, d_val, q_val in zip(
        feature_names, doc_vector, query_vector.toarray()[0]
    ):
        print("{:<15} {:<12.3f} {:<12.3f} {:<12.3f}".format(
            term, d_val, q_val, d_val * q_val
        ))

    norm_doc = np.sqrt(np.sum(doc_vector ** 2))
    norm_query = np.sqrt(np.sum(query_vector.toarray()[0] ** 2))

    print(
        f"Cosine Similarity = {cos_sim:.3f} | "
        f"Norm(Doc) = {norm_doc:.3f} | "
        f"Norm(Query) = {norm_query:.3f}"
    ) 
