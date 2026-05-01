import os, math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

ps = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


# ── PREPROCESSING ─────────────────────────────────────────────
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(w) for w in tokens if w.isalpha() and w not in STOP_WORDS]
    return tokens


# ── LOAD DOCUMENTS ────────────────────────────────────────────
def load_documents(folder):
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                docs[fname] = f.read()
    return docs


# ── TERM-DOCUMENT MATRIX ──────────────────────────────────────
def build_td_matrix(processed_docs):
    all_terms = sorted(set(t for tokens in processed_docs.values() for t in tokens))
    doc_names = list(processed_docs.keys())
    matrix = {}
    for term in all_terms:
        matrix[term] = {doc: (1 if term in processed_docs[doc] else 0)
                        for doc in doc_names}
    return matrix, all_terms, doc_names


def boolean_query_td(query, matrix, doc_names):
    """AND query over term-document matrix"""
    query_terms = preprocess(query)
    if not query_terms:
        return []
    # start with all docs, then AND each term
    result = set(doc_names)
    for term in query_terms:
        if term in matrix:
            matching = {doc for doc, val in matrix[term].items() if val == 1}
            result &= matching
        else:
            return []  # term not in any doc → no results
    return sorted(result)


# ── INVERTED INDEX ────────────────────────────────────────────
def build_inverted_index(processed_docs):
    index = {}
    for doc_name, tokens in processed_docs.items():
        for token in set(tokens):  # set to avoid duplicate postings
            index.setdefault(token, []).append(doc_name)
    return index


def boolean_query_inverted(query, index):
    """AND query over inverted index"""
    query_terms = preprocess(query)
    if not query_terms:
        return []
    # start with postings of first term
    result = set(index.get(query_terms[0], []))
    for term in query_terms[1:]:
        result &= set(index.get(term, []))
    return sorted(result)


# ── TF-IDF + COSINE SIMILARITY ────────────────────────────────
def build_tfidf(raw_docs):
    doc_names = list(raw_docs.keys())
    corpus = [raw_docs[d] for d in doc_names]
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, doc_names


def tfidf_query(query, vectorizer, matrix, doc_names, top_k=10):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    ranked_indices = np.argsort(-scores)
    results = [(doc_names[i], round(float(scores[i]), 4))
               for i in ranked_indices if scores[i] > 0]
    return results[:top_k]


# ── EVALUATION METRICS ────────────────────────────────────────
def precision(retrieved, relevant):
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(retrieved)


def recall(retrieved, relevant):
    if not relevant:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(relevant)


def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def average_precision(retrieved, relevant):
    relevant_set = set(relevant)
    score, hits = 0.0, 0
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            hits += 1
            score += hits / i
    return score / len(relevant) if relevant else 0.0