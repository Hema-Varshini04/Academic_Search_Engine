import math

# --------------------------------------------------
# DOCUMENT LINKS (REAL WEBSITES)
# --------------------------------------------------

doc_links = {
    "D1": "https://en.wikipedia.org/wiki/Web_mining",
    "D2": "https://en.wikipedia.org/wiki/Information_retrieval",
    "D3": "https://en.wikipedia.org/wiki/PageRank",
    "D4": "https://en.wikipedia.org/wiki/Search_engine"
}

# --------------------------------------------------
# DOCUMENT CONTENT
# --------------------------------------------------

documents = {
    "D1": "web mining and information retrieval",
    "D2": "information retrieval using search engines",
    "D3": "page rank algorithm in web search",
    "D4": "data mining techniques for web data"
}

# --------------------------------------------------
# HYPERLINK STRUCTURE
# --------------------------------------------------

links = {
    "D1": ["D2", "D4"],
    "D2": ["D3"],
    "D3": ["D4"],
    "D4": ["D2"]
}

# --------------------------------------------------
# DISPLAY DOCUMENT DATASET
# --------------------------------------------------

print("\nDOCUMENT DATASET")
print("-" * 70)
print("Doc\tLink\t\t\t\t\tWords")
print("-" * 70)

for doc in documents:
    print(f"{doc}\t{doc_links[doc]}\t{documents[doc]}")

# --------------------------------------------------
# QUERY INPUT
# --------------------------------------------------

query = input("\nEnter Query: ").lower().split()

N = len(documents)

tokens = {d: documents[d].lower().split() for d in documents}

# --------------------------------------------------
# TERM FREQUENCY
# --------------------------------------------------

print("\nTERM FREQUENCY (TF)")
print("Formula: TF = Term Count / Total Terms")
print("-" * 50)
print("Doc\tTerm\tCount\tTotal\tTF")

tf_table = {}

for doc in tokens:

    total = len(tokens[doc])
    tf_table[doc] = {}

    for term in query:

        count = tokens[doc].count(term)

        tf = count / total if total else 0

        tf_table[doc][term] = tf

        print(doc, term, count, total, round(tf,4))

# --------------------------------------------------
# INVERSE DOCUMENT FREQUENCY
# --------------------------------------------------

print("\nINVERSE DOCUMENT FREQUENCY (IDF)")
print("Formula: IDF = log(N / DF)")
print("-" * 40)
print("Term\tDF\tIDF")

idf_table = {}

for term in query:

    df = sum(1 for d in tokens if term in tokens[d])

    idf = math.log(N/df) if df else 0

    idf_table[term] = idf

    print(term, df, round(idf,4))

# --------------------------------------------------
# TF-IDF CALCULATION
# --------------------------------------------------

print("\nTF-IDF CALCULATION")
print("Formula: TF-IDF = TF × IDF")
print("-" * 40)
print("Doc\tTF-IDF Score")

tfidf_score = {}

for doc in tokens:

    score = 0

    for term in query:

        score += tf_table[doc][term] * idf_table[term]

    tfidf_score[doc] = score

    print(doc, round(score,4))

# --------------------------------------------------
# RELEVANCE CHECK
# --------------------------------------------------

if all(score == 0 for score in tfidf_score.values()):

    print("\nRESULT")
    print("-" * 40)
    print("All Documents are IRRELEVANT to the Query")
    exit()

# --------------------------------------------------
# PAGERANK CALCULATION
# --------------------------------------------------

print("\nPAGERANK CALCULATION")
print("Formula: PR(A) = (1-d)/N + d(PR(B)/L(B))")
print("d = 0.85")
print("-" * 50)

pagerank = {doc: 1/N for doc in documents}

d = 0.85

for i in range(5):

    new_pr = {}

    for doc in documents:

        rank_sum = 0

        for src in links:

            if doc in links[src]:

                rank_sum += pagerank[src] / len(links[src])

        new_pr[doc] = (1-d)/N + d*rank_sum

    pagerank = new_pr

print("Doc\tPageRank")

for doc in pagerank:

    print(doc, round(pagerank[doc],4))

# --------------------------------------------------
# FINAL SCORE
# --------------------------------------------------

print("\nFINAL RANKING SCORE")
print("Formula: Final Score = TF-IDF + PageRank")
print("-" * 50)

print("Doc\tTF-IDF\tPageRank\tFinal")

final_score = {}

for doc in documents:

    final = tfidf_score[doc] + pagerank[doc]

    final_score[doc] = final

    print(doc,
          round(tfidf_score[doc],4),
          round(pagerank[doc],4),
          round(final,4))

# --------------------------------------------------
# RANKED OUTPUT
# --------------------------------------------------

ranked = sorted(final_score.items(),
                key=lambda x: x[1],
                reverse=True)

print("\nRANKED DOCUMENTS")
print("-" * 30)

for i,(doc,score) in enumerate(ranked,1):

    print(i,".",doc,"->",round(score,4))