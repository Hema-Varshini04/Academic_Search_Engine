# Academic Search Engine

## 📌 Project Overview

The Academic Search Engine is a Flask-based web application that retrieves relevant Wikipedia documents based on a user query and ranks them using:

- TF-IDF (Term Frequency - Inverse Document Frequency)
- Cosine Similarity
- Vector Space Model

This project demonstrates fundamental Information Retrieval concepts.

---

## 🚀 Features

- Search academic topics dynamically
- Retrieve Wikipedia summaries
- Calculate TF, IDF, TF-IDF score
- Compute Cosine Similarity
- Rank documents based on relevance
- Modern dark themed user interface

---

## 🛠 Technologies Used

- Python
- Flask
- Scikit-learn
- Wikipedia API
- HTML & CSS

---

## 📂 Project Structure

Academic_Search_Engine/
│
├── app.py
├── requirements.txt
├── README.md
│
└── templates/
      └── index.html

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

git clone https://github.com/Hema-Varshini04/Academic_Search_Engine.git

cd Academic_Search_Engine

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run Application

python app.py

Open browser:

http://127.0.0.1:5000/

---

## 🧠 Working Principle

1. User enters a research query.
2. System fetches top 5 related Wikipedia topics.
3. TF-IDF vectorization is applied.
4. Cosine Similarity is calculated between query and documents.
5. Documents are ranked based on similarity score.
6. Results are displayed with TF, IDF, TF-IDF, and Cosine values.

---

## 📊 Output Includes

- Rank of document
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- TF-IDF Score
- Cosine Similarity Score
- Wikipedia URL
- Summary of document

---

## 👩‍💻 Developed For

Information Retrieval Assignment

---

## 📎 GitHub Repository

 https://github.com/Hema-Varshini04/Academic_Search_Engine.git

