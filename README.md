# TechWiki Smart Search Engine

**TechWiki** is an intelligent search engine that combines dictionary lookup, spell correction, and semantic search.  
The system automatically classifies queries into three modes:

- **Single word** → Dictionary lookup via **WordNet** → fallback to fuzzy search if no exact match.  
- **Two words** → Look up each word individually, combine results, and fallback to BM25 if no dictionary match.  
- **Three or more words** → Semantic search using **BM25** on a large dataset (Simple English Wikipedia).

---

## 📂 Project Structure
```
.
├── app.py                # Flask backend API
├── search_engine.py      # Search engine logic (dictionary + BM25)
├── index.html            # User interface (frontend)
├── requirments.txt       # Python dependencies
├── Simple_wiki_data.csv  # Search dataset (Simple English Wikipedia) (i zipped it)
├── bm25_model.pkl        # Pre-trained BM25 index base on csv data  for fast search (it will display when you run code)
├── file_id_SPW           # this is file id of SimpleWiki, you can take it on SimpleWiki
├── crawler_data.ipynb    # this is code to crawl data and clean it
```

---

## 🚀 Features
- **WordNet dictionary integration** for accurate word definitions, examples, synonyms, and antonyms.
- **Fuzzy search** for handling typos or misspellings.
- **BM25 ranking** for semantic document retrieval from a large corpus.
- **Stopword handling** to improve dictionary lookup accuracy.
- **Web interface** for easy searching.
- **Cached index** for fast startup with large datasets.

---

## 🛠 Installation
1. **Clone the repository**
```bash
git clone https://github.com/GiaoSuD/Smart-Search-Engine.git
cd <your-repo-name>
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**
```bash
pip install -r requirments.txt
```

4. **Download NLTK WordNet data**
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## ▶️ Usage
1. **Start the backend**
```bash
python app.py
```
Backend will run at: `http://127.0.0.1:5000`

2. **Open the frontend**  
Open `index.html` in your browser.

---

## 📌 API Endpoints
### `GET /search?query=<your_query>`
- Returns search results in JSON.
- Behavior:
  - **1 word** → Dictionary mode.
  - **2 words** → Dictionary for each word → fallback BM25.
  - **≥3 words** → BM25 search.

---

## 📊 Dataset
The dataset comes from **Simple English Wikipedia**, processed into:
- `merged_file_unique.csv` – content for BM25 search.
- `bm25_model.pkl` – pre-built BM25 index for faster search.
## Note
- If you clone this repo, pls change the path to csv file on you computer to run it work, the path in app.py.
- If you run the app.py for the first time, it's may take a while (about 2-3m) to run and create bm25_model.pkl because the data file is heavy 
