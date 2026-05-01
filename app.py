from flask import Flask, request, jsonify, render_template
from ir_engine import (load_documents, preprocess, build_td_matrix,
                       build_inverted_index, build_tfidf,
                       boolean_query_td, boolean_query_inverted,
                       tfidf_query, precision, recall, f1, average_precision)

app = Flask(__name__)

# ── Load and index once at startup ────────────────────────────
DOCS_FOLDER = 'documents'
raw_docs = load_documents(DOCS_FOLDER)
processed_docs = {name: preprocess(text) for name, text in raw_docs.items()}

td_matrix, all_terms, doc_names = build_td_matrix(processed_docs)
inv_index = build_inverted_index(processed_docs)
vectorizer, tfidf_matrix, tfidf_doc_names = build_tfidf(raw_docs)

print(f"Loaded {len(raw_docs)} documents.")


@app.route('/')
def index():
    return render_template('index.html', doc_count=len(raw_docs))


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    method = data.get('method', 'tfidf')
    # optional: user-provided relevant docs for eval
    relevant = data.get('relevant', [])

    if method == 'td_matrix':
        retrieved = boolean_query_td(query, td_matrix, doc_names)
        results = [{'doc': d, 'score': 1} for d in retrieved]

    elif method == 'inverted_index':
        retrieved = boolean_query_inverted(query, inv_index)
        results = [{'doc': d, 'score': 1} for d in retrieved]

    else:  # tfidf (default)
        tfidf_results = tfidf_query(
            query, vectorizer, tfidf_matrix, tfidf_doc_names)
        retrieved = [r[0] for r in tfidf_results]
        results = [{'doc': r[0], 'score': r[1]} for r in tfidf_results]

    # Evaluation (only if relevant docs are provided)
    eval_metrics = {}
    if relevant:
        p = precision(retrieved, relevant)
        r = recall(retrieved, relevant)
        eval_metrics = {
            'precision': round(p, 3),
            'recall': round(r, 3),
            'f1': round(f1(p, r), 3),
            'ap': round(average_precision(retrieved, relevant), 3)
        }

    return jsonify({'results': results, 'count': len(results), 'eval': eval_metrics})


@app.route('/document/<doc_name>')
def get_document(doc_name):
    if doc_name in raw_docs:
        return jsonify({'name': doc_name, 'content': raw_docs[doc_name][:1000]})
    return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
