from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]

        # Wikipedia Search
        topics = wikipedia.search(query, results=5)

        documents = []
        urls = []

        for topic in topics:
            try:
                page = wikipedia.page(topic)
                documents.append(page.summary)
                urls.append(page.url)
            except:
                pass

        if len(documents) == 0:
            return render_template("index.html", query=query)

        # TF-IDF Calculation
        vectorizer = TfidfVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(documents + [query])

        query_vector = tfidf_matrix[-1]
        document_vectors = tfidf_matrix[:-1]

        cosine_scores = cosine_similarity(query_vector, document_vectors)[0]

        results = []

        for i, doc in enumerate(documents):
            tfidf_values = document_vectors[i].toarray()[0]

            # Term Frequency (sum of tfidf values as approximation)
            tf = tfidf_values.sum()

            # Average IDF value
            idf = vectorizer.idf_.mean()

            # TF-IDF Score
            tfidf_score = tf * idf

            # Cosine Similarity
            cosine = cosine_scores[i]

            results.append({
                "rank": i + 1,
                "tf": round(tf, 4),
                "idf": round(idf, 4),
                "tfidf": round(tfidf_score, 4),
                "cosine": round(cosine, 4),
                "url": urls[i],
                "summary": doc[:500]
            })

        # Sort by Cosine Similarity
        results = sorted(results, key=lambda x: x["cosine"], reverse=True)[:5]

        return render_template("index.html", query=query, results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)