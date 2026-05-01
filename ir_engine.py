import os
import math
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',    quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ── Stemmers & Lemmatizer (instantiated once) ─────────────────
_porter = PorterStemmer()
_snowball = SnowballStemmer("english")
_lancaster = LancasterStemmer()
_lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


# ── WordNet POS helper for lemmatization ─────────────────────
def _get_wordnet_pos(word):
    from nltk import pos_tag
    tag = pos_tag([word])[0][1][0].upper()
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB,
               'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)


# ── Core preprocessor (returns tokens + trace) ───────────────
def preprocess(text, stemmer='porter', use_lemma=False, trace=False):
    """
    Parameters
    ----------
    text      : raw input string
    stemmer   : 'porter' | 'snowball' | 'lancaster'
    use_lemma : if True, lemmatize instead of stem
    trace     : if True, return dict with intermediate steps

    Returns
    -------
    list of processed tokens   (trace=False)
    dict with 'tokens', 'filtered', 'processed'  (trace=True)
    """
    # Step 1 — tokenise
    tokens = [w for w in word_tokenize(text.lower()) if w.isalpha()]

    # Step 2 — remove stop words
    filtered = [w for w in tokens if w not in STOP_WORDS]

    # Step 3 — stem or lemmatize
    if use_lemma:
        processed = [_lemmatizer.lemmatize(
            w, _get_wordnet_pos(w)) for w in filtered]
    else:
        stemmer_map = {
            'porter':   _porter.stem,
            'snowball': _snowball.stem,
            'lancaster': _lancaster.stem,
        }
        stem_fn = stemmer_map.get(stemmer, _porter.stem)
        processed = [stem_fn(w) for w in filtered]

    if trace:
        return {'tokens': tokens, 'filtered': filtered, 'processed': processed}
    return processed


# ── Load documents ────────────────────────────────────────────
def load_documents(folder):
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if fname.endswith('.txt'):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                docs[fname] = f.read()
    return docs


# ── Build indexes for a given preprocessing config ───────────
def build_indexes(raw_docs, stemmer='porter', use_lemma=False):
    """Returns (processed_docs, td_matrix, all_terms, doc_names, inv_index)"""
    processed_docs = {
        name: preprocess(text, stemmer=stemmer, use_lemma=use_lemma)
        for name, text in raw_docs.items()
    }
    doc_names = list(processed_docs.keys())
    all_terms = sorted(set(t for tokens in processed_docs.values()
                       for t in tokens))

    # Term-Document Matrix
    td_matrix = {
        term: {
            doc: (1 if term in processed_docs[doc] else 0) for doc in doc_names}
        for term in all_terms
    }

    # Inverted Index
    inv_index = {}
    for doc_name, tokens in processed_docs.items():
        for token in set(tokens):
            inv_index.setdefault(token, []).append(doc_name)

    return processed_docs, td_matrix, all_terms, doc_names, inv_index


# ── Build TF-IDF index ────────────────────────────────────────
def build_tfidf(raw_docs, stemmer='porter', use_lemma=False):
    doc_names = list(raw_docs.keys())
    corpus = [
        ' '.join(preprocess(raw_docs[d], stemmer=stemmer, use_lemma=use_lemma))
        for d in doc_names
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix, doc_names


# ══════════════════════════════════════════════════════════════
# BOOLEAN QUERY PARSER  (AND / OR / NOT)
# ══════════════════════════════════════════════════════════════

def _parse_boolean_query(query_str, stemmer='porter', use_lemma=False):
    """
    Parse a query string into a list of (operator, term) tuples.
    Supports:  AND  OR  NOT  (case-insensitive)

    Examples
    --------
    "space AND nasa"          → [(AND, 'space'), (AND, 'nasa')]
    "space OR mars"           → [(AND, 'space'), (OR,  'mars')]
    "space NOT military"      → [(AND, 'space'), (NOT, 'militari')]
    "hockey OR baseball NOT violence" → mixed
    """
    tokens = query_str.strip().split()
    parts = []        # (operator, raw_word)
    i = 0
    pending_op = 'AND'

    while i < len(tokens):
        tok = tokens[i].upper()
        if tok in ('AND', 'OR', 'NOT'):
            pending_op = tok
            i += 1
        else:
            parts.append((pending_op, tokens[i]))
            pending_op = 'AND'      # default between consecutive words
            i += 1

    # Stem / lemmatize each word
    result = []
    for op, word in parts:
        processed = preprocess(word, stemmer=stemmer, use_lemma=use_lemma)
        if processed:
            result.append((op, processed[0]))
    return result


def boolean_query_td(query_str, td_matrix, doc_names,
                     stemmer='porter', use_lemma=False):
    """Boolean retrieval over Term-Document Matrix. Supports AND/OR/NOT."""
    parsed = _parse_boolean_query(
        query_str, stemmer=stemmer, use_lemma=use_lemma)
    if not parsed:
        return []

    result_set = None   # will be set on first AND/initial term

    for op, term in parsed:
        matching = set(
            doc for doc in doc_names
            if td_matrix.get(term, {}).get(doc, 0) == 1
        )
        if result_set is None or op == 'AND':
            result_set = matching if result_set is None else result_set & matching
        elif op == 'OR':
            result_set = result_set | matching
        elif op == 'NOT':
            result_set = result_set - matching

    return sorted(result_set) if result_set else []


def boolean_query_inverted(query_str, inv_index, doc_names,
                           stemmer='porter', use_lemma=False):
    """Boolean retrieval over Inverted Index. Supports AND/OR/NOT."""
    parsed = _parse_boolean_query(
        query_str, stemmer=stemmer, use_lemma=use_lemma)
    if not parsed:
        return []

    result_set = None

    for op, term in parsed:
        postings = set(inv_index.get(term, []))
        if result_set is None or op == 'AND':
            result_set = postings if result_set is None else result_set & postings
        elif op == 'OR':
            result_set = result_set | postings
        elif op == 'NOT':
            result_set = result_set - postings

    return sorted(result_set) if result_set else []


# ══════════════════════════════════════════════════════════════
# TF-IDF RANKED RETRIEVAL
# ══════════════════════════════════════════════════════════════

def tfidf_query(query, vectorizer, matrix, doc_names,
                stemmer='porter', use_lemma=False, top_k=10):
    processed_query = ' '.join(preprocess(
        query, stemmer=stemmer, use_lemma=use_lemma))
    q_vec = vectorizer.transform([processed_query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    ranked = np.argsort(-scores)
    return [
        (doc_names[i], round(float(scores[i]), 4))
        for i in ranked if scores[i] > 0
    ][:top_k]


# ══════════════════════════════════════════════════════════════
# TF-IDF TERM WEIGHTS FOR A SINGLE DOCUMENT
# ══════════════════════════════════════════════════════════════

def get_doc_tfidf_weights(doc_name, doc_names, vectorizer, matrix, top_n=10):
    """Return top-N TF-IDF weighted terms for a specific document."""
    if doc_name not in doc_names:
        return []
    idx = doc_names.index(doc_name)
    feature_names = vectorizer.get_feature_names_out()
    scores = matrix[idx].toarray().flatten()
    ranked = np.argsort(-scores)
    return [
        {'term': feature_names[i], 'score': round(float(scores[i]), 4)}
        for i in ranked[:top_n] if scores[i] > 0
    ]


# ══════════════════════════════════════════════════════════════
# VOCABULARY / IDF STATISTICS
# ══════════════════════════════════════════════════════════════

def get_vocabulary_stats(raw_docs, vectorizer, matrix, doc_names,
                         stemmer='porter', use_lemma=False):
    """Return corpus-level statistics for the statistics page."""
    feature_names = vectorizer.get_feature_names_out()
    n_docs = len(doc_names)
    n_terms = len(feature_names)

    # IDF values from the fitted vectorizer
    idf_values = vectorizer.idf_
    # Highest IDF = most unique/specific terms
    top_idf_idx = np.argsort(-idf_values)[:15]
    # Lowest IDF  = most common terms
    low_idf_idx = np.argsort(idf_values)[:15]

    # Average TF-IDF per term across all docs
    avg_tfidf = np.asarray(matrix.mean(axis=0)).flatten()
    top_avg_idx = np.argsort(-avg_tfidf)[:15]

    return {
        'n_docs':   n_docs,
        'n_terms':  int(n_terms),
        'most_specific': [
            {'term': feature_names[i], 'idf': round(float(idf_values[i]), 4)}
            for i in top_idf_idx
        ],
        'most_common': [
            {'term': feature_names[i], 'idf': round(float(idf_values[i]), 4)}
            for i in low_idf_idx
        ],
        'highest_tfidf': [
            {'term': feature_names[i], 'score': round(float(avg_tfidf[i]), 4)}
            for i in top_avg_idx
        ],
    }


# ══════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════

def precision(retrieved, relevant):
    if not retrieved:
        return 0.0
    return round(len(set(retrieved) & set(relevant)) / len(retrieved), 4)


def recall(retrieved, relevant):
    if not relevant:
        return 0.0
    return round(len(set(retrieved) & set(relevant)) / len(relevant), 4)


def f1(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def average_precision(retrieved, relevant):
    relevant_set = set(relevant)
    score, hits = 0.0, 0
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant_set:
            hits += 1
            score += hits / i
    return round(score / len(relevant), 4) if relevant else 0.0


def mean_average_precision(queries_results):
    """
    queries_results : list of dicts
        [{'retrieved': [...], 'relevant': [...]}, ...]
    Returns MAP score.
    """
    if not queries_results:
        return 0.0
    ap_scores = [
        average_precision(q['retrieved'], q['relevant'])
        for q in queries_results
    ]
    return round(sum(ap_scores) / len(ap_scores), 4)


# ══════════════════════════════════════════════════════════════
# PREDEFINED TEST SUITE  (for MAP evaluation — from Sec 6)
# ══════════════════════════════════════════════════════════════

# These are representative queries per domain with hand-picked
# relevant document name substrings for automatic matching.
TEST_SUITE = [
    {'query': 'surgery patient',         'domain': 'medicine',
        'keywords': ['medicine']},
    {'query': 'space exploration rocket',
        'domain': 'space',          'keywords': ['space']},
    {'query': 'machine learning neural',  'domain': 'computer_science',
        'keywords': ['computer_science']},
    {'query': 'hockey baseball game',
        'domain': 'sports',         'keywords': ['sports']},
    {'query': 'democracy election vote',
        'domain': 'politics',       'keywords': ['politics']},
]


def build_test_relevant(doc_names):
    """
    Auto-build relevant sets for the test suite by matching
    doc filename keywords to domain labels.
    """
    suite = []
    for test in TEST_SUITE:
        relevant = [
            d for d in doc_names
            if any(kw in d.lower() for kw in test['keywords'])
        ]
        suite.append({
            'query':    test['query'],
            'domain':   test['domain'],
            'relevant': relevant,
        })
    return suite

