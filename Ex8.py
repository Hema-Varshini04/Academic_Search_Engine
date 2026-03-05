import requests
from bs4 import BeautifulSoup
import math
import re

# Step 1: URLs (Medical Domain)
urls = [
    "https://en.wikipedia.org/wiki/Diabetes",
    "https://en.wikipedia.org/wiki/Heart_disease",
    "https://en.wikipedia.org/wiki/Cancer",
    "https://en.wikipedia.org/wiki/Obesity",
    "https://en.wikipedia.org/wiki/Vaccination"
]

documents = []
headers = {"User-Agent": "Mozilla/5.0"}

# Step 2: Crawl and fetch documents
for url in urls:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p")[:3])
    documents.append(text)

# Display fetched documents
print("FETCHED DOCUMENTS:\n")
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(doc[:300], "...")
    print("-" * 60)

# Step 3: Input search query
query = input("Enter search term: ")

# Step 4: Preprocessing
stop_words = {"the","is","a","an","and","to","in","of","can","like","that","for","on","with","as","by"}
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

processed_docs = [preprocess(doc) for doc in documents]
processed_query = preprocess(query)
N = len(documents)

# 🔹 TOTAL WORDS IN EACH DOCUMENT
doc_word_counts = [len(doc) for doc in processed_docs]
print("\nTOTAL WORDS IN EACH DOCUMENT:")
for i, count in enumerate(doc_word_counts):
    print(f"Document {i+1}: {count} words")

# Step 5: Term–Document Frequency Table
freq_table = []
for word in processed_query:
    freqs = [doc.count(word) for doc in processed_docs]
    freq_table.append([word] + freqs)

print("\nTERM–DOCUMENT FREQUENCY TABLE")
header = ["Term"] + [f"Doc{i+1}" for i in range(N)]
print("{:<15} ".format("Term") + " ".join("{:<10}".format(h) for h in header[1:]))

for row in freq_table:
    print("{:<15} ".format(row[0]) + " ".join("{:<10}".format(f) for f in row[1:]))

# Step 6: TF Calculation
tf_table = []
for row in freq_table:
    tf_vals = []
    for i, freq in enumerate(row[1:]):
        total_words = doc_word_counts[i]
        tf = freq / total_words if total_words > 0 else 0
        tf_vals.append(tf)
    tf_table.append([row[0]] + tf_vals)

print("\nTERM FREQUENCY (TF) TABLE")
print("{:<15} ".format("Term") + " ".join("{:<10}".format(h) for h in header[1:]))

for row in tf_table:
    print("{:<15} ".format(row[0]) + " ".join("{:<10.4f}".format(f) for f in row[1:]))

# Step 7: IDF Calculation
idf_values = []
for row in freq_table:
    df = sum(1 for f in row[1:] if f > 0)
    idf = math.log(N / df) if df != 0 else 0
    idf_values.append([row[0], df, idf])

print("\nINVERSE DOCUMENT FREQUENCY (IDF) TABLE")
print("{:<15} {:<10} {:<10}".format("Term", "DF", "IDF"))

for row in idf_values:
    print("{:<15} {:<10} {:<10.4f}".format(*row))

# Step 8: TF-IDF Calculation
tfidf_table = []
for i, row in enumerate(tf_table):
    tfidf_vals = [tf_val * idf_values[i][2] for tf_val in row[1:]]
    tfidf_table.append([row[0]] + tfidf_vals)

print("\nTF-IDF TABLE")
print("{:<15} ".format("Term") + " ".join("{:<10}".format(h) for h in header[1:]))

for row in tfidf_table:
    print("{:<15} ".format(row[0]) + " ".join("{:<10.6f}".format(f) for f in row[1:]))

# Step 9: TF-IDF Sum and Ranking
tfidf_sum = [0.0 for _ in range(N)]

for row in tfidf_table:
    for i in range(N):
        tfidf_sum[i] += row[i+1]

print("\nTF-IDF SCORE PER DOCUMENT:")
for i, score in enumerate(tfidf_sum):
    print(f"Document {i+1}: {score:.6f}")

best = tfidf_sum.index(max(tfidf_sum))
print("\nMOST RELEVANT DOCUMENT:")
print(f"Document {best+1}")
print(urls[best])

