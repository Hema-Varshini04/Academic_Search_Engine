import requests
from bs4 import BeautifulSoup
import math

# Step 1: URLs
urls = [
 "https://en.wikipedia.org/wiki/Search_engine",
 "https://en.wikipedia.org/wiki/Text_mining",
 "https://en.wikipedia.org/wiki/Natural_language_processing"

]

documents = []
headers = {"User-Agent": "Mozilla/5.0"}

# Step 2: Fetch documents
for url in urls:
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(p.text.lower() for p in soup.find_all("p")[:3])
    documents.append(text)

print("FETCHED DOCUMENTS:\n")
for i, doc in enumerate(documents):
    print(f"Document {i+1}:")
    print(doc[:200], "...")
    print("-" * 60)

# TOTAL WORD COUNT
print("\nTOTAL WORDS IN EACH DOCUMENT:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {len(doc.split())} words")

# Step 3: Input query
query = input("Enter search term: ").lower()
query_words = query.split()
N = len(documents)

# Step 4: Term–Document Frequency Table
freq_table = []

for word in query_words:
    freqs = [doc.split().count(word) for doc in documents]
    freq_table.append([word] + freqs)

print("\nTERM–DOCUMENT FREQUENCY TABLE")
header = ["Term"] + [f"Doc{i+1}" for i in range(N)]
print("{:<15} {:<10} {:<10} {:<10}".format(*header))

for row in freq_table:
    print("{:<15} {:<10} {:<10} {:<10}".format(*row))

# Step 5: TF Calculation
tf_table = []

for row in freq_table:
    word = row[0]
    tf_values = []

    for i, freq in enumerate(row[1:]):
        total_words = len(documents[i].split())
        tf = freq / total_words
        tf_values.append(tf)

    tf_table.append([word] + tf_values)

print("\nTERM FREQUENCY (TF) TABLE : TF = freq / total_words")
print("{:<15} {:<10} {:<10} {:<10}".format(*header))

for row in tf_table:
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f}".format(*row))

# Step 6: IDF Calculation
idf_values = []

for row in freq_table:
    df = sum(1 for f in row[1:] if f > 0)
    idf = math.log(N / df) if df != 0 else 0
    idf_values.append([row[0], df, idf])

print("\nINVERSE DOCUMENT FREQUENCY (IDF) TABLE: IDF = log(N / DF)")
print("{:<15} {:<10} {:<10}".format("Term", "DF", "IDF"))

for row in idf_values:
    print("{:<15} {:<10} {:<10.4f}".format(*row))

# Step 7: TF-IDF Calculation
tfidf_table = []

for i, row in enumerate(tf_table):
    word = row[0]
    tfidf_values = [tf * idf_values[i][2] for tf in row[1:]]
    tfidf_table.append([word] + tfidf_values)

print("\nTF-IDF TABLE: TF-IDF = TF X IDF")
print("{:<15} {:<10} {:<10} {:<10}".format(*header))

for row in tfidf_table:
    print("{:<15} {:<10.6f} {:<10.6f} {:<10.6f}".format(*row))

# Step 8: Ranking
tfidf_sum = [0.0 for _ in range(N)]

for row in tfidf_table:
    for i in range(N):
        tfidf_sum[i] += row[i + 1]

print("\nTF-IDF SCORE:")
for i, score in enumerate(tfidf_sum):
    print(f"Document {i+1}: {score:.6f}")

best = tfidf_sum.index(max(tfidf_sum))

print("\nMOST RELEVANT DOCUMENT:")
print(f"Document {best+1}")
print(urls[best])