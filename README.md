# Retrivo — Information Retrieval System

> A full-pipeline IR system built as a course project for the Information Retrieval course at the Faculty of Computers and Information, Minia University.

🌐 **Live Demo:** [https://retrivoapp.dpdns.org](https://retrivoapp.dpdns.org)

---

## Overview

Retrivo is a complete, web-based Information Retrieval system that demonstrates the full IR pipeline — from raw text preprocessing through indexing, ranked retrieval, and quantitative evaluation. It operates on a curated corpus of 100 Wikipedia articles spanning five knowledge domains and exposes three independent retrieval models through a modern, responsive web interface.

---

## Features

### Preprocessing Pipeline
- **Tokenization** — splits raw text into word tokens using NLTK, discarding punctuation and numeric tokens
- **Stop Word Removal** — filters 179 common English stop words using NLTK's built-in list
- **Stemming** — three interchangeable algorithms: Porter, Snowball (English), and Lancaster
- **Lemmatization** — WordNet-based lemmatizer with POS tagging for accurate root forms
- **Live Preprocessing Trace** — every search shows the query passing step-by-step through the pipeline, plus a side-by-side comparison of all four stemmers/lemmatizer

### Retrieval Models
| Model | Type | Operators |
|---|---|---|
| Term-Document Matrix | Boolean | AND, OR, NOT |
| Inverted Index | Boolean | AND, OR, NOT |
| TF-IDF + Cosine Similarity | Ranked | Free text |

All three models support **Boolean operators** (AND / OR / NOT) in the query for the Boolean methods, and all share the same preprocessing pipeline.

### Evaluation Metrics
- **Precision** — fraction of retrieved documents that are relevant
- **Recall** — fraction of relevant documents that were retrieved
- **F1-Score** — harmonic mean of precision and recall
- **Average Precision (AP)** — ranking-aware metric rewarding early relevant results
- **Mean Average Precision (MAP)** — automatic batch evaluation across 5 predefined domain queries with known relevant document sets

### Vocabulary & IDF Statistics Page
- Total document and vocabulary counts
- Top 15 most specific terms (highest IDF)
- Top 15 most common terms (lowest IDF)
- Top 15 most discriminative terms (highest average TF-IDF)
- Switchable between all three stemmers

### Document Preview Modal
- First 1,200 characters of any retrieved document
- Top 10 TF-IDF weighted terms for that document with proportional bar visualization

---

## Dataset

The corpus consists of **100 Wikipedia article summaries** across five knowledge domains (20 articles per domain):

| Domain | Sample Topics |
|---|---|
| Medicine | Surgery, Pharmacology, Cardiology, Cancer, Vaccine, Diabetes |
| Space | Astronomy, NASA, Black Hole, Mars, Rocket, Hubble Telescope |
| Computer Science | AI, Machine Learning, Deep Learning, NLP, Cryptography, IR |
| Sports | Football, Basketball, Tennis, Ice Hockey, Olympic Games |
| Politics | Democracy, Election, Parliament, Human Rights, Treaty |

Articles are fetched using the `wikipedia-api` Python library and saved as plain `.txt` files in the `documents/` directory.

---

## Project Structure

```
retrivoapp/
├── app.py              # Flask web server — HTTP endpoints
├── ir_engine.py        # All IR logic — preprocessing, indexing, retrieval, evaluation
├── requirements.txt    # Python dependencies
├── documents/          # Corpus — 100 Wikipedia .txt files (not tracked by git)
└── templates/
    └── index.html      # Frontend — single responsive HTML page
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the main web interface |
| POST | `/search` | Runs a query — returns ranked results + evaluation metrics |
| POST | `/preprocess` | Returns preprocessing trace and stemmer comparison for a query |
| GET | `/document/<name>` | Returns document preview + TF-IDF term weights |
| GET | `/stats` | Returns vocabulary and IDF statistics |
| POST | `/evaluate_map` | Runs MAP evaluation across 5 predefined test queries |

### `/search` Request Body
```json
{
  "query": "space exploration",
  "method": "tfidf",
  "stemmer": "porter",
  "use_lemma": false,
  "relevant": ["doc_046_space_NASA.txt"]
}
```
`method` options: `tfidf` · `td_matrix` · `inverted_index`
`stemmer` options: `porter` · `snowball` · `lancaster`

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/MohamedELfaidy/IRSection-Project.git
cd IRSection-Project

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
"

# 5. Generate the document corpus
python generate_dataset.py

# 6. Run the development server
python app.py
# Open http://localhost:8001
```

---

## Deployment

The application is deployed at **[https://retrivoapp.dpdns.org](https://retrivoapp.dpdns.org)** using:
- **Gunicorn** — production WSGI server (2 workers, port 8001)
- **Nginx** — reverse proxy with HTTPS termination
- **Let's Encrypt** — TLS certificate
- **systemd** — process management with automatic restart

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| flask | 3.0.0 | Web framework |
| gunicorn | 21.2.0 | Production WSGI server |
| nltk | 3.8.1 | Tokenization, stop words, stemming, lemmatization |
| scikit-learn | 1.4.2 | TF-IDF vectorization, cosine similarity |
| numpy | ≥1.26, <2 | Vector operations |
| pandas | 2.2.0 | Data handling |
| wikipedia-api | 0.6.0 | Dataset generation |
| beautifulsoup4 | 4.12.3 | Web crawling (Section 4 feature) |

---

## Academic Context

This project was developed as part of the **Information Retrieval** course at **Minia University, Faculty of Computers and Information**.

The implementation covers all six course lab sections:

| Section | Topic | Implemented In |
|---|---|---|
| 1 | Tokenization, stop word removal | `ir_engine.preprocess()` |
| 2 | Porter, Snowball, Lancaster stemming; WordNet lemmatization | `ir_engine.preprocess()` with stemmer selector |
| 3 | Term-Document Matrix, Inverted Index, Boolean queries | `ir_engine.boolean_query_td/inverted()` |
| 4 | Web crawling with BeautifulSoup | Dataset pipeline |
| 5 | TF-IDF, cosine similarity, document ranking | `ir_engine.tfidf_query()` |
| 6 | Precision, Recall, F1, AP, MAP | `ir_engine.precision/recall/f1/average_precision/mean_average_precision()` |

**Under supervision of:**
- Dr. Ebtissam AbdelHakam — Course Instructor
- Eng. Marco Edwar — Teaching Assistant

**Project Team:**
1. Mohamed Sayed Saad Mojawer
2. Ammar Yasser AbdEllatif AbdElAzim
3. Beshoy Farouk Gaber
4. Hussien Mohamed Haggag
5. Michael Hany Kamal

---

## License

This project is submitted as academic coursework. All rights reserved by the project team.
