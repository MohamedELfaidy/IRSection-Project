from flask import Flask, request, jsonify, render_template
from ir_engine import (
    load_documents, preprocess, build_indexes, build_tfidf,
    boolean_query_td, boolean_query_inverted, tfidf_query,
    get_doc_tfidf_weights, get_vocabulary_stats,
    precision, recall, f1, average_precision, mean_average_precision,
    build_test_relevant
)

app = Flask(__name__)

# ── Load & index everything once at startup ───────────────────
DOCS_FOLDER = 'documents'
raw_docs = load_documents(DOCS_FOLDER)
print(f"Loaded {len(raw_docs)} documents.")

# Default indexes (Porter stemmer)
processed_docs, td_matrix, all_terms, doc_names, inv_index = \
    build_indexes(raw_docs, stemmer='porter', use_lemma=False)

vectorizer, tfidf_matrix, tfidf_doc_names = \
    build_tfidf(raw_docs, stemmer='porter', use_lemma=False)

# Pre-build test suite for MAP
test_suite = build_test_relevant(doc_names)

# Cache for other stemmer/lemma combos so we don't rebuild on
# every request — keyed by (stemmer, use_lemma)
_index_cache = {}
_tfidf_cache = {}


def get_indexes(stemmer, use_lemma):
    key = (stemmer, use_lemma)
    if key not in _index_cache:
        pd, tdm, terms, dn, inv = build_indexes(
            raw_docs, stemmer=stemmer, use_lemma=use_lemma)
        _index_cache[key] = (pd, tdm, terms, dn, inv)
    return _index_cache[key]


def get_tfidf(stemmer, use_lemma):
    key = (stemmer, use_lemma)
    if key not in _tfidf_cache:
        vec, mat, dn = build_tfidf(
            raw_docs, stemmer=stemmer, use_lemma=use_lemma)
        _tfidf_cache[key] = (vec, mat, dn)
    return _tfidf_cache[key]


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', doc_count=len(raw_docs))


# ── /preprocess  — show pipeline steps for a query ───────────
@app.route('/preprocess', methods=['POST'])
def preprocess_route():
    data = request.json
    query = data.get('query', '')
    stemmer = data.get('stemmer', 'porter')
    use_lemma = data.get('use_lemma', False)

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    trace = preprocess(query, stemmer=stemmer,
                       use_lemma=use_lemma, trace=True)

    # Also show all three stemmers + lemma side by side
    comparison = {}
    for s in ('porter', 'snowball', 'lancaster'):
        comparison[s] = preprocess(query, stemmer=s, use_lemma=False)
    comparison['lemmatizer'] = preprocess(query, use_lemma=True)

    return jsonify({'trace': trace, 'comparison': comparison})


# ── /search  — main retrieval endpoint ───────────────────────
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'tfidf')
    stemmer = data.get('stemmer', 'porter')
    use_lemma = data.get('use_lemma', False)
    relevant = data.get('relevant', [])

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    retrieved = []
    results = []

    if method == 'td_matrix':
        pd, tdm, terms, dn, inv = get_indexes(stemmer, use_lemma)
        retrieved = boolean_query_td(
            query, tdm, dn, stemmer=stemmer, use_lemma=use_lemma)
        results = [{'doc': d, 'score': 1} for d in retrieved]

    elif method == 'inverted_index':
        pd, tdm, terms, dn, inv = get_indexes(stemmer, use_lemma)
        retrieved = boolean_query_inverted(
            query, inv, dn, stemmer=stemmer, use_lemma=use_lemma)
        results = [{'doc': d, 'score': 1} for d in retrieved]

    else:  # tfidf (default)
        vec, mat, dn = get_tfidf(stemmer, use_lemma)
        tfidf_results = tfidf_query(
            query, vec, mat, dn, stemmer=stemmer, use_lemma=use_lemma)
        retrieved = [r[0] for r in tfidf_results]
        results = [{'doc': r[0], 'score': r[1]} for r in tfidf_results]

    # Preprocessing trace for the query
    trace = preprocess(query, stemmer=stemmer,
                       use_lemma=use_lemma, trace=True)

    # Evaluation
    eval_metrics = {}
    if relevant:
        p = precision(retrieved, relevant)
        r = recall(retrieved, relevant)
        eval_metrics = {
            'precision': p,
            'recall':    r,
            'f1':        f1(p, r),
            'ap':        average_precision(retrieved, relevant),
        }

    return jsonify({
        'results': results,
        'count':   len(results),
        'eval':    eval_metrics,
        'trace':   trace,
    })


# ── /document/<name>  — preview document text ────────────────
@app.route('/document/<doc_name>')
def get_document(doc_name):
    if doc_name not in raw_docs:
        return jsonify({'error': 'Not found'}), 404

    # TF-IDF weights for this document
    weights = get_doc_tfidf_weights(
        doc_name, tfidf_doc_names, vectorizer, tfidf_matrix, top_n=10)

    return jsonify({
        'name':    doc_name,
        'content': raw_docs[doc_name][:1200],
        'weights': weights,
    })


@app.route('/documents_list')
def documents_list():
    return jsonify({
        'documents': list(raw_docs.keys()),
        'count': len(raw_docs)
    })

# ── /stats  — vocabulary & IDF statistics ────────────────────
@app.route('/stats')
def stats():
    stemmer = request.args.get('stemmer', 'porter')
    use_lemma = request.args.get('use_lemma', 'false').lower() == 'true'
    vec, mat, dn = get_tfidf(stemmer, use_lemma)
    data = get_vocabulary_stats(raw_docs, vec, mat, dn,
                                stemmer=stemmer, use_lemma=use_lemma)
    return jsonify(data)


# ── /evaluate_map  — MAP across predefined test suite ────────
@app.route('/evaluate_map', methods=['POST'])
def evaluate_map():
    data = request.json or {}
    stemmer = data.get('stemmer', 'porter')
    use_lemma = data.get('use_lemma', False)

    vec, mat, dn = get_tfidf(stemmer, use_lemma)
    suite = build_test_relevant(dn)

    results_per_query = []
    for test in suite:
        tfidf_res = tfidf_query(
            test['query'], vec, mat, dn,
            stemmer=stemmer, use_lemma=use_lemma, top_k=20)
        retrieved = [r[0] for r in tfidf_res]
        p = precision(retrieved, test['relevant'])
        r = recall(retrieved, test['relevant'])
        ap = average_precision(retrieved, test['relevant'])
        results_per_query.append({
            'query':     test['query'],
            'domain':    test['domain'],
            'retrieved': retrieved[:5],
            'relevant_count': len(test['relevant']),
            'precision': p,
            'recall':    r,
            'f1':        f1(p, r),
            'ap':        ap,
        })

    map_score = mean_average_precision([
        {'retrieved': [r for r in item['retrieved']],
         'relevant':  build_test_relevant(dn)[i]['relevant']}
        for i, item in enumerate(results_per_query)
    ])

    return jsonify({'map': map_score, 'queries': results_per_query})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8001)
