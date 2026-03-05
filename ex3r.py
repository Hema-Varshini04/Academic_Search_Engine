# ==========================================
# INDEXING & SEARCHING – QUERY BASED EXECUTION
# DOCUMENTS READ FROM FOLDER
# ==========================================

import os
import re
from collections import defaultdict

# ------------------------------------------
# STEP 0: LOAD DOCUMENTS FROM FOLDER
# ------------------------------------------
DOCUMENT_FOLDER = "documents"

documents = {}
doc_ids = []

for i, filename in enumerate(sorted(os.listdir(DOCUMENT_FOLDER)), start=1):
    if filename.endswith(".txt"):
        doc_id = f"D{i}"
        doc_ids.append(doc_id)
        with open(os.path.join(DOCUMENT_FOLDER, filename), "r", encoding="utf-8") as f:
            documents[doc_id] = f.read().strip()

# ------------------------------------------
# PREPROCESSING (TOKENIZATION)
# ------------------------------------------
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())

tokenized_docs = {doc_id: tokenize(text) for doc_id, text in documents.items()}

# ------------------------------------------
# PRE-COMPUTATIONS
# ------------------------------------------
vocabulary = sorted(set(word for tokens in tokenized_docs.values() for word in tokens))

inverted_index = defaultdict(list)
for doc_id, tokens in tokenized_docs.items():
    for token in set(tokens):
        inverted_index[token].append(doc_id)

word_positions = defaultdict(lambda: defaultdict(list))
for doc_id, tokens in tokenized_docs.items():
    for pos, token in enumerate(tokens):   # 0-based (corrected)
        word_positions[token][doc_id].append(pos)

# ------------------------------------------
# INDEX DISPLAY FUNCTIONS
# ------------------------------------------
def show_tokenization():
    for doc_id, tokens in tokenized_docs.items():
        print(f"{doc_id}: {', '.join(tokens)}")

def show_vocabulary():
    for word in vocabulary:
        print(word)

def show_term_document_matrix():
    header = f"{'Term':<15}"
    for doc_id in doc_ids:
        header += f"{doc_id:<4}"
    print(header)
    print("-" * (15 + 4 * len(doc_ids)))

    for term in vocabulary:
        row = f"{term:<15}"
        for doc_id in doc_ids:
            row += f"{'✔' if term in tokenized_docs[doc_id] else '':<4}"
        print(row)

def show_occurrence_list():
    for term in vocabulary:
        print(f"{term:<15} Doc IDs: {', '.join(inverted_index[term])}")

def show_word_positions():
    for term in vocabulary:
        postings = []
        for doc_id, positions in word_positions[term].items():
            postings.append(f"{doc_id}:{positions}")
        print(f"{term:<15} {', '.join(postings)}")

# ------------------------------------------
# COMPUTATIONAL SEARCH ENGINE FUNCTIONS
# ------------------------------------------
def single_term_search():
    term = input("Enter term: ").lower()
    if term in inverted_index:
        print(f"Term '{term}' found in Doc ID(s): {', '.join(inverted_index[term])}")
    else:
        print("Term not found.")

def prefix_search():
    query = input("Enter prefix query (end with *): ").lower()

    if not query.endswith("*"):
        print("Invalid prefix query. Use * at the end.")
        return

    prefix = query[:-1]
    matches = [t for t in vocabulary if t.startswith(prefix)]

    if not matches:
        print("No matching terms.")
        return

    for term in matches:
        print(f"Term '{term}' → Doc ID(s): {', '.join(inverted_index[term])}")

def range_search():
    start = input("Enter start term: ").lower()
    end = input("Enter end term: ").lower()

    terms = [t for t in vocabulary if start <= t <= end]
    if not terms:
        print("No terms in range.")
        return

    result_docs = set()
    for term in terms:
        result_docs.update(inverted_index[term])

    print("Terms in range:", ", ".join(terms))
    print("Doc ID(s):", ", ".join(sorted(result_docs)))

def context_search():
    query = input("Enter term, prefix*, or two-word phrase: ").lower().strip()
    parts = query.split()

    # ----------------------------------
    # PREFIX CONTEXT
    # ----------------------------------
    if len(parts) == 1 and parts[0].endswith("*"):
        prefix = parts[0][:-1]
        matched = [t for t in vocabulary if t.startswith(prefix)]

        if not matched:
            print("No matching terms found.")
            return

        for term in matched:
            print(f"\nContext for term '{term}':")
            for doc_id, positions in word_positions[term].items():
                tokens = tokenized_docs[doc_id]
                print(f"  Doc ID {doc_id}:")
                for pos in positions:
                    start = max(0, pos - 2)
                    end = min(len(tokens), pos + 3)
                    print("    ...", " ".join(tokens[start:end]), "...")

    # ----------------------------------
    # SINGLE TERM CONTEXT
    # ----------------------------------
    elif len(parts) == 1:
        term = parts[0]
        if term not in word_positions:
            print("Term not found.")
            return

        print(f"\nContext for term '{term}':")
        for doc_id, positions in word_positions[term].items():
            tokens = tokenized_docs[doc_id]
            print(f"  Doc ID {doc_id}:")
            for pos in positions:
                start = max(0, pos - 2)
                end = min(len(tokens), pos + 3)
                print("    ...", " ".join(tokens[start:end]), "...")

    # ----------------------------------
    # PHRASE CONTEXT (2 WORDS)
    # ----------------------------------
    elif len(parts) == 2:
        t1, t2 = parts
        found = False

        for doc_id, tokens in tokenized_docs.items():
            for i in range(len(tokens) - 1):
                if tokens[i] == t1 and tokens[i + 1] == t2:
                    found = True
                    start = max(0, i - 2)
                    end = min(len(tokens), i + 4)
                    print(f"\nDoc ID {doc_id}:")
                    print("    ...", " ".join(tokens[start:end]), "...")

        if not found:
            print("Phrase not found.")

    else:
        print("Only single word, prefix*, or two-word phrase allowed.")

# ------------------------------------------
# QUERY MENU
# ------------------------------------------
while True:
    print("\n========= QUERY MENU =========")
    print("1. Show Tokenization")
    print("2. Show Vocabulary")
    print("3. Show Term–Document Matrix")
    print("4. Show Occurrence List")
    print("5. Show Word Position Inverted Index")
    print("6. Computational Search Engine")
    print("0. Exit")

    choice = input("Enter query number: ").strip()

    if choice == "1":
        show_tokenization()
    elif choice == "2":
        show_vocabulary()
    elif choice == "3":
        show_term_document_matrix()
    elif choice == "4":
        show_occurrence_list()
    elif choice == "5":
        show_word_positions()
    elif choice == "6":
        while True:
            print("\n===== COMPUTATIONAL SEARCH ENGINE =====")
            print("1. Single")
            print("2. Prefix")
            print("3. Range")
            print("4. Context")
            print("0. Exit")

            c = input("Enter choice: ").strip()

            if c == "1":
                single_term_search()
            elif c == "2":
                prefix_search()
            elif c == "3":
                range_search()
            elif c == "4":
                context_search()
            elif c == "0":
                break
            else:
                print("Invalid choice.")

    elif choice == "0":
        print("Exiting program.")
        break
    else:
        print("Invalid query number.")