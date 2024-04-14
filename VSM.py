import os
import math
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Preprocess and tokenize text
def clean_and_tokenize_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    return filtered_tokens

# Compute term frequency (TF)
def compute_term_frequencies(tokens):
    frequencies = defaultdict(int)
    for token in tokens:
        frequencies[token] += 1
    return dict(frequencies)

# Compute inverse document frequency (IDF)
def compute_inverse_document_frequency(doc_tokens):
    document_count = defaultdict(int)
    total_docs = len(doc_tokens)
    
    for tokens in doc_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            document_count[term] += 1
    
    idf_scores = {}
    for term, count in document_count.items():
        idf_scores[term] = math.log(total_docs / count)
    
    return idf_scores

# Compute TF-IDF scores
def compute_tf_idf_scores(tf_scores, idf_scores):
    tf_idf_scores = {}
    for term, count in tf_scores.items():
        tf_idf_scores[term] = count * idf_scores.get(term, 0)
    return tf_idf_scores

# Calculate the cosine similarity
def compute_cosine_similarity(vec1, vec2):
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in vec1)
    magnitude1 = np.sqrt(sum(value ** 2 for value in vec1.values()))
    magnitude2 = np.sqrt(sum(value ** 2 for value in vec2.values()))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# Retrieve relevant documents based on the query
@app.route('/search', methods=['POST'])
def search_documents():
    input_query = request.json.get('query', '')
    if not input_query:
        return jsonify([])

    query_tokens = clean_and_tokenize_text(input_query)
    query_tf = compute_term_frequencies(query_tokens)
    query_tf_idf = compute_tf_idf_scores(query_tf, overall_idf)
    
    document_scores = []
    for index, doc_tf_idf in enumerate(documents_tf_idf):
        similarity = compute_cosine_similarity(query_tf_idf, doc_tf_idf)
        if similarity >= 0.025:  # Filter very low scores
            document_scores.append((index + 1, similarity))
    
    document_scores.sort(key=lambda x: x[1], reverse=True)
    return jsonify(document_scores)

# Initialize document data
document_tokens = []
term_freq_list = []

for i in range(1, 27):
    filepath = f"d:/deskto/ASSIGNMENTS/IR A2/ResearchPapers/{i}.txt"
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            tokens = clean_and_tokenize_text(file.read())
            document_tokens.append(tokens)
            term_freq_list.append(compute_term_frequencies(tokens))

overall_idf = compute_inverse_document_frequency(document_tokens)
documents_tf_idf = [compute_tf_idf_scores(tf, overall_idf) for tf in term_freq_list]

if __name__ == "__main__":
    app.run(debug=True)
