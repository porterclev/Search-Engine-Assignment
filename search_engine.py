#-------------------------------------------------------------
# AUTHOR: Porter Cleivdence
# FILENAME: search_engine.
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
# --> add your Python code here

print(documents)

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------
# --> add your Python code here

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=lambda text: [PorterStemmer().stem(token) for token in text.lower().split()],
    stop_words='english',
    ngram_range=(1, 2),
    binary=True
)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------
# --> add your Python code here

vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
# --> add your Python code here

query = "I love a dog"
query_vector = vectorizer.transform([query])

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here

doc_vectors = document_matrix.toarray()
query_vector = query_vector.toarray()

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------

scores = []
# --> add your Python code here
for doc_index, doc in enumerate(doc_vectors):
    dot_prd = 0
    for i, e in enumerate(doc):
        dot_prod += e * query_vector[0][i]

    scores.append((dot_prod, documents[doc_index]))


# for score in scores:
#   print(score)
# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------

ranking = []
ranking = sorted(scores, key=lambda x: x[0], reverse=True)
for score, doc in ranking:
    print(score, "->", doc)