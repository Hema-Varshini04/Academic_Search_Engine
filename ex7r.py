# ---------------------------------------
# N-gram Based Spell Detection & Correction
# (WITH STEP-BY-STEP TABLE OUTPUT)
# ---------------------------------------

# MODULE 1: USER INPUT
word = input("Enter a word: ").lower().strip()

# MODULE 2: N-GRAM GENERATION  (BIGRAMS — like your image)
def generate_ngrams(text, n=2):   # changed to BIGRAM (image output uses bigrams)
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngrams.add(text[i:i+n])
    return ngrams

# MODULE 3: DICTIONARY
# (same module — only words adjusted to match image example)
dictionary = [
    "data",
    "database",
    "datamining",
    "mining",
    "machine"
]

# Generate n-grams for input word
input_ngrams = generate_ngrams(word)

print("\nOUTPUT:\n")
print("MISSPELLED WORD :", word)
print("\nBigrams of Misspelled Word :", input_ngrams)

print("\n" + "="*95)
print("TABLE SHOWING INTERSECTION (I), UNION (U) AND JACCARD CALCULATION")
print("="*95)

# MODULE 4 & 5: MATCHING + SIMILARITY (JACCARD)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(union) == 0:
        return 0, intersection, union

    similarity = len(intersection) / len(union)

    return similarity, intersection, union


# TABLE HEADER
print(f"\n{'Word':12} {'Intersection (I)':30} {'Union (U)':35} {'|I|':4} {'|U|':4} {'Calculation':15} {'Similarity'}")
print("-"*120)

# MODULE 6: CANDIDATE RANKING
scores = []

for dict_word in dictionary:

    dict_ngrams = generate_ngrams(dict_word)

    similarity, intersection, union = jaccard_similarity(
        input_ngrams,
        dict_ngrams
    )

    scores.append((dict_word, similarity))

    calculation = f"{len(intersection)}/{len(union)}"

    print(f"{dict_word:12} "
          f"{str(intersection):30} "
          f"{str(union):35} "
          f"{len(intersection):4} "
          f"{len(union):4} "
          f"{calculation:15} "
          f"{round(similarity,2)}")

# sort ranking
scores.sort(key=lambda x: x[1], reverse=True)

# MODULE 7: BEST CORRECTION SELECTION
best_match = scores[0] if scores[0][1] > 0 else None

print("\n" + "-"*95)

print("\nINTERSECTION (I) : Common bigrams between misspelled word and dictionary word")
print("UNION (U)        : All unique bigrams from both words")
print("FORMULA USED     : Jaccard Similarity = |I| / |U|")

# MODULE 8: OUTPUT DISPLAY
print("\n--- SPELL CHECK RESULT ---")
print("Input Word :", word)

if best_match:
    print("Correct Word :", best_match[0])
    print("Similarity Score :", round(best_match[1], 3))
else:
    print("No suitable correction found")