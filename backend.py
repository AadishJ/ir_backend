import re
import os
import math
import nltk
import fuzzy
import pickle
import unicodedata
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------
# Setup
# ----------------------------
nltk.download("punkt")
nltk.download("stopwords")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
alpha_pattern = re.compile(r"^[a-zA-Z]+$")
soundex = fuzzy.Soundex(4)
spell = SpellChecker()


def normalize_ascii(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if ord(c) < 128)


def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if alpha_pattern.match(t) and len(t) > 1]
    tokens = [t for t in tokens if t not in stop_words]

    corrected_tokens = []
    cache = {}
    misspelled = spell.unknown(tokens)

    for t in tokens:
        if t in cache:
            corrected_tokens.append(cache[t])
        elif t in misspelled:
            corrected = spell.correction(t)
            corrected = corrected if corrected else t
            cache[t] = corrected
            corrected_tokens.append(corrected)
        else:
            cache[t] = t
            corrected_tokens.append(t)

    stems = [stemmer.stem(t) for t in corrected_tokens]

    soundex_codes = []
    for token in stems:
        try:
            clean_token = re.sub(r"[^a-zA-Z]", "", token)
            if clean_token:
                ascii_token = normalize_ascii(clean_token)
                soundex_codes.append(soundex(ascii_token))
            else:
                soundex_codes.append("")
        except Exception:
            soundex_codes.append("")

    return stems, soundex_codes


def build_index(results):
    dictionary = {}
    doc_lengths = {}
    N = len(results)

    for docID, data in enumerate(results.values(), start=1):
        tokens = data["tokens"]
        tf_counts = Counter(tokens)

        weights = {}
        for term, tf in tf_counts.items():
            w = 1 + math.log10(tf)
            weights[term] = w

            if term not in dictionary:
                dictionary[term] = {"df": 0, "postings": []}

            dictionary[term]["df"] += 1
            dictionary[term]["postings"].append((docID, tf))

        length = math.sqrt(sum(w**2 for w in weights.values()))
        doc_lengths[docID] = length

    return dictionary, doc_lengths, N


def process_query(query, dictionary, doc_lengths, N):
    stems, _ = preprocess_text(query)
    original_tokens = [
        t
        for t in nltk.word_tokenize(query.lower())
        if alpha_pattern.match(t) and len(t) > 1 and t not in stop_words
    ]

    expanded_tokens = []
    for t in stems:
        if t not in dictionary:
            code = soundex(normalize_ascii(t))
            soundex_matches = [
                term for term in dictionary if soundex(normalize_ascii(term)) == code
            ]
            expanded_tokens.extend(soundex_matches if soundex_matches else [t])
        else:
            expanded_tokens.append(t)

    tf_counts = Counter(expanded_tokens)
    query_weights = {}

    for term, tf in tf_counts.items():
        if term in dictionary:
            idf = math.log10(N / dictionary[term]["df"])
            w = (1 + math.log10(tf)) * idf
            query_weights[term] = w

    norm = math.sqrt(sum(w**2 for w in query_weights.values()))
    if norm > 0:
        for term in query_weights:
            query_weights[term] /= norm

    return query_weights, original_tokens, stems


def rank_documents(query_weights, dictionary, doc_lengths, top_k=10):
    scores = defaultdict(float)

    for term, wq in query_weights.items():
        if term not in dictionary:
            continue
        postings = dictionary[term]["postings"]
        for docID, tf in postings:
            wd = 1 + math.log10(tf)
            scores[docID] += wq * (wd / doc_lengths[docID])

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


# ----------------------------
# Flask API
# ----------------------------
app = Flask(__name__)
CORS(app)

# Load index and filenames at startup
index_file = "index.pkl"
corpus_dir = "Corpus"

if os.path.exists(index_file):
    with open(index_file, "rb") as f:
        dictionary, doc_lengths, N, filenames = pickle.load(f)
else:
    raise RuntimeError("Index file not found. Please build the index first.")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_weights, query_terms, query_stems = process_query(
        query, dictionary, doc_lengths, N
    )
    ranked_docs = rank_documents(query_weights, dictionary, doc_lengths)

    results = []
    for docID, score in ranked_docs:
        fname = filenames[docID - 1]
        filepath = os.path.join(corpus_dir, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            text = ""
        results.append(
            {"doc_id": docID, "filename": fname, "score": score, "text": text}
        )

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
