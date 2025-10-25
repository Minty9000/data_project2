import os, re, math
from collections import Counter

# ----------------------------- PREPROCESSING -----------------------------

def read_doc_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def clean_text(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower().strip()

def remove_stopwords(text, stopwords):
    words = [w for w in text.split() if w and w not in stopwords]
    return ' '.join(words)

def stem_and_lemmatize(text):
    words = []
    for w in text.split():
        if w.endswith("ing") and len(w) > 4:
            w = w[:-3]
        elif w.endswith("ly") and len(w) > 3:
            w = w[:-2]
        elif w.endswith("ment") and len(w) > 5:
            w = w[:-4]
        words.append(w)
    return ' '.join(words)

def preprocess_document(doc_path, stopwords):
    with open(doc_path, 'r') as f:
        text = f.read()
    text = clean_text(text)
    text = remove_stopwords(text, stopwords)
    text = stem_and_lemmatize(text)
    output_name = "preproc_" + os.path.basename(doc_path)
    with open(output_name, 'w') as f:
        f.write(text)
    print(f"✅ Preprocessed → {output_name}")
    return output_name  # return new file name

# ----------------------------- TF-IDF -----------------------------

def compute_tf(word_counts):
    total = sum(word_counts.values())
    return {w: c / total for w, c in word_counts.items()}

def compute_idf(all_docs):
    num_docs = len(all_docs)
    word_doc_count = Counter()
    for words in all_docs.values():
        for w in set(words):
            word_doc_count[w] += 1
    return {w: math.log(num_docs / word_doc_count[w]) + 1 for w in word_doc_count}

def compute_tfidf(all_docs):
    idf = compute_idf(all_docs)
    results = {}
    for doc, words in all_docs.items():
        tf = compute_tf(Counter(words))
        scores = {w: round(tf[w] * idf[w], 2) for w in tf}
        sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:5]
        results[doc] = sorted_scores
    return results

# ----------------------------- MAIN PIPELINE -----------------------------

def main():
    # Automatically use these files (no user input)
    doc_list_file = "tfidf_docs.txt"
    stopwords_file = "stopwords.txt"

    print(f"📁 Using document list: {doc_list_file}")
    print(f"📁 Using stopwords file: {stopwords_file}")

    # Load stopwords
    with open(stopwords_file, 'r') as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())

    # Load document names from tfidf_docs.txt
    docs = read_doc_list(doc_list_file)

    # Step 1: Preprocess all docs
    preproc_docs = [preprocess_document(d, stopwords) for d in docs]

    # Step 2: Compute TF-IDF
    all_docs = {}
    for d in preproc_docs:
        with open(d, 'r') as f:
            all_docs[d] = f.read().split()

    tfidf_results = compute_tfidf(all_docs)

    # Step 3: Write outputs
    for d, results in tfidf_results.items():
        out_name = "tfidf_" + d.replace("preproc_", "")
        with open(out_name, 'w') as f:
            f.write(str(results))
        print(f"✅ TF-IDF written to {out_name}")

    print("\n🎯 All TF-IDF computations complete.")


if __name__ == "__main__":
    main()
