import wikipedia
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
query=input("Enter your query: ").lower()
query=re.sub(r'[^a-zA-Z ]','',query)
stop_words=set(ENGLISH_STOP_WORDS)
query_words=[w for w in query.split() if w not in stop_words]
print("\nQuery After Stopword Removal:",query_words)
search_results=wikipedia.search(query,results=2)
documents=[]
urls=[]
for title in search_results:
    try:
        page=wikipedia.page(title,auto_suggest=False)
        text=page.content.lower()
        text=re.sub(r'[^a-zA-Z ]','',text)
        documents.append(text)
        urls.append(page.url)
    except:
        continue
print("\nFetched URLs:")
for i,url in enumerate(urls):
    print(f"D{i+1} → {url}")
print("\n========== DOCUMENT PREVIEW (200 WORDS) ==========")
for i,doc in enumerate(documents):
    preview=" ".join(doc.split()[:200])
    print(f"\nD{i+1} Preview:\n{preview}")
clean_docs=[]
for doc in documents:
    words=[w for w in doc.split() if w not in stop_words]
    clean_docs.append(words)
vocabulary=list(set(query_words))
N=len(clean_docs)
print("\nVocabulary (Query Words Only):",vocabulary)
print("\n========== DF TABLE ==========")
print("{:<15} {:<5}".format("TERM","DF"))
print("-"*25)
df_list=[]
for term in vocabulary:
    df=sum(1 for doc in clean_docs if term in doc)
    df_list.append(df)
    print("{:<15} {:<5}".format(term,df))
print("\n========== NORMALIZED TF ==========")
tf=[]
for d_index,doc in enumerate(clean_docs):
    doc_len=len(doc)
    row=[]
    print(f"\nDocument D{d_index+1}")
    print("{:<15} {:<10}".format("TERM","TF"))
    print("-"*25)
    for term in vocabulary:
        value=doc.count(term)/doc_len
        row.append(value)
        print("{:<15} {:.6f}".format(term,value))
    tf.append(row)
print("\n========== IDF TABLE ==========")
print("{:<15} {:<10}".format("TERM","IDF"))
print("-"*25)
idf=[]
for i,term in enumerate(vocabulary):
    idf_value=math.log((N+1)/(df_list[i]+1))+1
    idf.append(idf_value)
    print("{:<15} {:.6f}".format(term,idf_value))
tfidf=[]
print("\n========== TF-IDF TABLE ==========")
for d_index,row in enumerate(tf):
    print(f"\nDocument D{d_index+1}")
    print("{:<15} {:<10}".format("TERM","TF-IDF"))
    print("-"*25)
    tfidf_row=[]
    for i in range(len(vocabulary)):
        value=row[i]*idf[i]
        tfidf_row.append(value)
        print("{:<15} {:.6f}".format(vocabulary[i],value))
    tfidf.append(tfidf_row)
query_tf=[]
for term in vocabulary:
    query_tf.append(query_words.count(term)/len(query_words))
query_vector=np.array([query_tf[i]*idf[i] for i in range(len(vocabulary))]).reshape(1,-1)
doc_matrix=np.array(tfidf)
similarity=cosine_similarity(doc_matrix,query_vector)
print("\n========== COSINE SIMILARITY ==========")
scores=[]
for i,score in enumerate(similarity):
    print(f"D{i+1} : {score[0]:.6f}")
    scores.append(score[0])
print("\n========== RANKING ==========")
ranked=sorted([(i,score) for i,score in enumerate(scores)],key=lambda x:x[1],reverse=True)
for rank,(doc_id,score) in enumerate(ranked,1):
    print(f"Rank {rank} → D{doc_id+1} (Score = {score:.6f})")
max_score=ranked[0][1]
print("\n========== MOST RELEVANT DOCUMENT(S) ==========")
for doc_id,score in ranked:
    if score==max_score:
        print(f"D{doc_id+1} → {urls[doc_id]}")