from flask import Flask, request, jsonify
from flask_cors import CORS
import csv
import json
import re
import pickle
import os
from collections import defaultdict, Counter
import math
from difflib import get_close_matches
from rank_bm25 import BM25Okapi
import time
from datetime import datetime
from nltk.corpus import wordnet as wn
import nltk
csv.field_size_limit(2**20)

# Modified NLTK data download with better error handling
def initialize_nltk():
    try:
        # Download required NLTK data
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Test WordNet functionality
        test_word = 'test'
        wn.synsets(test_word)
        print("NLTK WordNet initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing NLTK WordNet: {e}")
        return False

# Kiểm tra và tải dữ liệu WordNet nếu chưa có
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

from nltk.corpus import wordnet
app = Flask(__name__)
CORS(app)

class SmartSearchEngine:
    def __init__(self):
        self.dictionary_data = {}
        self.csv_documents = []
        self.preprocessed_docs = []
        self.bm25 = None
        self.bm25_cache_file = 'bm25_model.pkl'
        self.dictionary_cache_file = 'dictionary_cache.pkl'
        self.vocabulary = set()
        self.inverted_index = defaultdict(list)
        self.use_wordnet = True  # Flag to enable/disable WordNet
        
    def load_dictionary(self, dictionary_file):
        """Load English dictionary from CSV or JSON"""
        print(f"Loading dictionary from {dictionary_file}...")
        
        try:
            if dictionary_file.endswith('.json'):
                with open(dictionary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # List of dictionaries
                        for item in data:
                            if isinstance(item, dict) and 'word' in item:
                                word = item['word'].lower().strip()
                                definition = item.get('definition', item.get('meaning', ''))
                                self.dictionary_data[word] = {
                                    'word': item['word'],
                                    'definition': definition,
                                    'type': item.get('type', item.get('part_of_speech', 'unknown')),
                                    'example': item.get('example', ''),
                                    'source': 'dictionary'
                                }
                    elif isinstance(data, dict):
                        # Dictionary format
                        for word, info in data.items():
                            word_key = word.lower().strip()
                            if isinstance(info, str):
                                self.dictionary_data[word_key] = {
                                    'word': word,
                                    'definition': info,
                                    'type': 'unknown',
                                    'example': '',
                                    'source': 'dictionary'
                                }
                            elif isinstance(info, dict):
                                self.dictionary_data[word_key] = {
                                    'word': word,
                                    'definition': info.get('definition', info.get('meaning', '')),
                                    'type': info.get('type', info.get('part_of_speech', 'unknown')),
                                    'example': info.get('example', ''),
                                    'source': 'dictionary'
                                }
                                
            elif dictionary_file.endswith('.csv'):
                with open(dictionary_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try different common column names
                        word = None
                        definition = None
                        
                        # Common word column names
                        for col in ['word', 'term', 'vocabulary', 'lexeme']:
                            if col in row and row[col]:
                                word = row[col].lower().strip()
                                break
                        
                        # Common definition column names
                        for col in ['definition', 'meaning', 'description', 'explanation']:
                            if col in row and row[col]:
                                definition = row[col]
                                break
                        
                        if word and definition:
                            self.dictionary_data[word] = {
                                'word': row.get('word', word),
                                'definition': definition,
                                'type': row.get('type', row.get('part_of_speech', row.get('pos', 'unknown'))),
                                'example': row.get('example', row.get('sample', '')),
                                'source': 'dictionary'
                            }
            
            print(f"Loaded {len(self.dictionary_data)} dictionary entries")
            
        except FileNotFoundError:
            print(f"Dictionary file {dictionary_file} not found. Creating sample dictionary...")
            self.create_sample_dictionary()
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            self.create_sample_dictionary()
    
    def create_sample_dictionary(self):
        """Create sample dictionary data"""
        sample_dict = {
            'python': {
                'word': 'Python',
                'definition': 'A high-level programming language known for its simplicity and readability.',
                'type': 'noun',
                'example': 'Python is widely used for web development and data science.',
                'source': 'dictionary'
            },
            'algorithm': {
                'word': 'Algorithm',
                'definition': 'A step-by-step procedure for solving a problem or completing a task.',
                'type': 'noun',
                'example': 'The sorting algorithm efficiently organized the data.',
                'source': 'dictionary'
            },
            'machine': {
                'word': 'Machine',
                'definition': 'A device that uses power to perform work or carry out tasks.',
                'type': 'noun',
                'example': 'The machine automated the manufacturing process.',
                'source': 'dictionary'
            },
            'learning': {
                'word': 'Learning',
                'definition': 'The process of acquiring knowledge, skills, or understanding.',
                'type': 'noun',
                'example': 'Continuous learning is essential for personal growth.',
                'source': 'dictionary'
            },
            'database': {
                'word': 'Database',
                'definition': 'An organized collection of structured information stored electronically.',
                'type': 'noun',
                'example': 'The database stores customer information securely.',
                'source': 'dictionary'
            },
            'network': {
                'word': 'Network',
                'definition': 'A group of interconnected computers or systems that share resources.',
                'type': 'noun',
                'example': 'The office network allows file sharing between computers.',
                'source': 'dictionary'
            },
            'security': {
                'word': 'Security',
                'definition': 'Protection against threats, vulnerabilities, or unauthorized access.',
                'type': 'noun',
                'example': 'Cybersecurity is crucial for protecting digital assets.',
                'source': 'dictionary'
            },
            'cloud': {
                'word': 'Cloud',
                'definition': 'Computing services delivered over the internet.',
                'type': 'noun',
                'example': 'Cloud storage allows access to files from anywhere.',
                'source': 'dictionary'
            },
            'artificial': {
                'word': 'Artificial',
                'definition': 'Made or produced by human beings rather than occurring naturally.',
                'type': 'adjective',
                'example': 'Artificial intelligence mimics human cognitive functions.',
                'source': 'dictionary'
            },
            'intelligence': {
                'word': 'Intelligence',
                'definition': 'The ability to acquire, understand, and use knowledge and skills.',
                'type': 'noun',
                'example': 'Human intelligence involves reasoning and problem-solving.',
                'source': 'dictionary'
            }
        }
        
        self.dictionary_data = sample_dict
        print(f"Created sample dictionary with {len(sample_dict)} entries")
    
    def preprocess_text(self, text):
        """Clean and tokenize text for BM25"""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their'
        }
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
        return tokens
    
    def load_csv_documents(self, csv_file):
        """Load documents from CSV for BM25 search"""
        print(f"Loading CSV documents from {csv_file}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc = {
                        'id': len(self.csv_documents),
                        'original_id': row.get('ID', row.get('id', str(len(self.csv_documents)))),
                        'title': row.get('Title', row.get('title', '')),
                        'content': row.get('Content', row.get('content', row.get('text', ''))),
                        'source': 'csv'
                    }
                    self.csv_documents.append(doc)
                    
                    # Preprocess for BM25
                    full_text = f"{doc['title']} {doc['content']}"
                    tokens = self.preprocess_text(full_text)
                    self.preprocessed_docs.append(tokens)
                    
                    # Build vocabulary
                    self.vocabulary.update(tokens)
            
            print(f"Loaded {len(self.csv_documents)} CSV documents")
            
        except FileNotFoundError:
            print(f"CSV file {csv_file} not found. Creating sample CSV data...")
            self.create_sample_csv_data()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.create_sample_csv_data()
    
    def create_sample_csv_data(self):
        """Create sample CSV data"""
        sample_docs = [
            {
                'original_id': '1',
                'title': 'Python Programming Fundamentals',
                'content': 'Python is a versatile programming language that emphasizes code readability and simplicity. It supports multiple programming paradigms including object-oriented, functional, and procedural programming. Python is widely used in web development, data science, artificial intelligence, and automation.',
                'source': 'csv'
            },
            {
                'original_id': '2', 
                'title': 'Machine Learning Algorithms',
                'content': 'Machine learning algorithms enable computers to learn and make decisions from data without being explicitly programmed. Common algorithms include linear regression, decision trees, neural networks, and support vector machines. These algorithms are used in recommendation systems, image recognition, and predictive analytics.',
                'source': 'csv'
            },
            {
                'original_id': '3',
                'title': 'Database Design Principles',
                'content': 'Database design involves organizing data efficiently to minimize redundancy and ensure data integrity. Key principles include normalization, establishing relationships between tables, and creating appropriate indexes. Well-designed databases improve performance and maintainability of applications.',
                'source': 'csv'
            },
            {
                'original_id': '4',
                'title': 'Network Security Fundamentals',
                'content': 'Network security protects data during transmission and prevents unauthorized access to network resources. It includes firewalls, encryption, intrusion detection systems, and access control mechanisms. Proper network security is essential for protecting sensitive information and maintaining business continuity.',
                'source': 'csv'
            },
            {
                'original_id': '5',
                'title': 'Cloud Computing Services',
                'content': 'Cloud computing provides on-demand access to computing resources over the internet. It includes Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). Cloud services offer scalability, cost-effectiveness, and accessibility from anywhere.',
                'source': 'csv'
            }
        ]
        
        for i, doc_data in enumerate(sample_docs):
            doc = {
                'id': i,
                'original_id': doc_data['original_id'],
                'title': doc_data['title'],
                'content': doc_data['content'],
                'source': 'csv'
            }
            self.csv_documents.append(doc)
            
            # Preprocess for BM25
            full_text = f"{doc['title']} {doc['content']}"
            tokens = self.preprocess_text(full_text)
            self.preprocessed_docs.append(tokens)
            
            # Build vocabulary
            self.vocabulary.update(tokens)
    
    def initialize_bm25(self, force_rebuild=False):
        """Initialize or load cached BM25 model"""
        if not force_rebuild and os.path.exists(self.bm25_cache_file):
            try:
                print("Loading cached BM25 model...")
                with open(self.bm25_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.bm25 = cache_data['bm25']
                    self.preprocessed_docs = cache_data['preprocessed_docs']
                    self.vocabulary = cache_data['vocabulary']
                    print(f"Loaded cached BM25 model with {len(self.preprocessed_docs)} documents")
                return
            except Exception as e:
                print(f"Error loading BM25 cache: {e}")
        
        print("Building new BM25 model...")
        if self.preprocessed_docs:
            self.bm25 = BM25Okapi(self.preprocessed_docs)
            
            # Cache the model
            try:
                cache_data = {
                    'bm25': self.bm25,
                    'preprocessed_docs': self.preprocessed_docs,
                    'vocabulary': self.vocabulary,
                    'timestamp': datetime.now().isoformat()
                }
                with open(self.bm25_cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"BM25 model cached to {self.bm25_cache_file}")
            except Exception as e:
                print(f"Error caching BM25 model: {e}")
        else:
            print("No documents available for BM25")
    
    def exact_dictionary_lookup(self, word):
        """Exact lookup in dictionary"""
        word_key = word.lower().strip()
        return self.dictionary_data.get(word_key)
    
    def fuzzy_dictionary_search(self, word, max_results=5):
        """Fuzzy search with spell correction"""
        word_key = word.lower().strip()
        
        # Get close matches
        matches = get_close_matches(word_key, self.dictionary_data.keys(), 
                                   n=max_results, cutoff=0.6)
        
        results = []
        for match in matches:
            entry = self.dictionary_data[match]
            similarity = self.calculate_similarity(word_key, match)
            results.append({
                **entry,
                'similarity_score': similarity,
                'search_type': 'fuzzy'
            })
        
        return results
    
    def calculate_similarity(self, word1, word2):
        """Calculate similarity score between two words"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, word1, word2).ratio()
    
    def bm25_search(self, query, limit=10):
        """BM25 search for multi-word queries"""
        if not self.bm25:
            return []
        
        query_tokens = self.preprocess_text(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        scored_docs = [(i, scores[i]) for i in range(len(scores))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scored_docs[:limit]:
            if score > 0:
                doc = self.csv_documents[doc_id]
                snippet = self.create_snippet(doc['content'], query_tokens)
                results.append({
                    'id': doc['original_id'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'snippet': snippet,
                    'score': round(score, 4),
                    'search_type': 'bm25',
                    'source': doc['source']
                })
        
        return results
    
    def create_snippet(self, content, query_tokens, max_length=200):
        """Create snippet highlighting query terms"""
        if not content:
            return ""
        
        content_lower = content.lower()
        
        # Find best position to start snippet
        best_pos = 0
        for token in query_tokens:
            pos = content_lower.find(token.lower())
            if pos != -1:
                best_pos = max(0, pos - 30)
                break
        
        # Extract snippet
        snippet = content[best_pos:best_pos + max_length]
        
        # Clean boundaries
        if best_pos > 0:
            space_pos = snippet.find(' ')
            if space_pos != -1:
                snippet = snippet[space_pos + 1:]
            snippet = "..." + snippet
        
        if len(content) > best_pos + max_length:
            last_space = snippet.rfind(' ')
            if last_space != -1:
                snippet = snippet[:last_space]
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def get_wordnet_info(self, word):
        """Get word information from WordNet with improved error handling"""
        if not self.use_wordnet:
            return None
            
        try:
            word = word.lower().strip()
            synsets = wn.synsets(word)
            
            if not synsets:
                return None
                
            word_info = {
                'word': word,
                'definitions': [],
                'examples': [],
                'synonyms': set(),
                'antonyms': set(),
                'pos_tags': set()
            }
            
            for synset in synsets:
                # Add definition
                word_info['definitions'].append({
                    'definition': synset.definition(),
                    'pos': synset.pos()
                })
                
                # Add examples
                word_info['examples'].extend(synset.examples())
                
                # Add synonyms and antonyms
                for lemma in synset.lemmas():
                    word_info['synonyms'].add(lemma.name())
                    if lemma.antonyms():
                        word_info['antonyms'].add(lemma.antonyms()[0].name())
                
                word_info['pos_tags'].add(synset.pos())
            
            # Convert sets to lists for JSON serialization
            word_info['synonyms'] = list(word_info['synonyms'])
            word_info['antonyms'] = list(word_info['antonyms'])
            word_info['pos_tags'] = list(word_info['pos_tags'])
            
            return word_info
            
        except Exception as e:
            print(f"WordNet lookup error for word '{word}': {e}")
            return None

    def smart_search(self, query):
        """Main search function with improved error handling"""
        query = query.strip()
        if not query:
            return {'error': 'Empty query', 'results': []}
        
        words = query.split()
        word_count = len(words)
        
        start_time = time.time()
        
        if word_count == 1:
            word = words[0]
            
            # Try WordNet lookup with fallback
            if self.use_wordnet:
                try:
                    wordnet_result = self.get_wordnet_info(word)
                    if wordnet_result:
                        search_time = round((time.time() - start_time) * 1000, 2)
                        return {
                            'query': query,
                            'search_type': 'wordnet',
                            'word_count': word_count,
                            'results': [{
                                **wordnet_result,
                                'search_type': 'wordnet',
                                'source': 'wordnet'
                            }],
                            'search_time_ms': search_time,
                            'total_results': 1
                        }
                except Exception as e:
                    print(f"WordNet search failed: {e}")
                    # Continue with fallback methods
        
            # Fallback to dictionary lookup
            exact_result = self.exact_dictionary_lookup(word)
            if exact_result:
                search_time = round((time.time() - start_time) * 1000, 2)
                return {
                    'query': query,
                    'search_type': 'exact_dictionary',
                    'word_count': word_count,
                    'results': [dict(exact_result, search_type='exact')],
                    'search_time_ms': search_time,
                    'total_results': 1
                }
            
            # Continue with existing fuzzy search
            fuzzy_results = self.fuzzy_dictionary_search(word)
            search_time = round((time.time() - start_time) * 1000, 2)
            return {
                'query': query,
                'search_type': 'fuzzy_dictionary',
                'word_count': word_count,
                'results': fuzzy_results,
                'search_time_ms': search_time,
                'total_results': len(fuzzy_results)
            }
        
        elif word_count == 2:
            # Check if first word is stopword
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
            }
    
            results = []
            if words[0].lower() in stopwords:
                # Khi từ đầu là stopword (the, a, an...), tìm kiếm từ thứ hai
                word = words[1]
        
                # Thử tìm kiếm chính xác trong từ điển
                exact_result = self.exact_dictionary_lookup(word)
                if exact_result:
                    search_time = round((time.time() - start_time) * 1000, 2)
                    return {
                        'query': query, 
                        'search_type': 'exact_dictionary',
                        'word_count': 1,
                        'results': [dict(exact_result, search_type='exact')],
                        'search_time_ms': search_time,
                        'total_results': 1
                    }
            
                # Nếu không có kết quả chính xác, thử tìm kiếm WordNet
                if self.use_wordnet:
                    try:
                        wordnet_result = self.get_wordnet_info(word)
                        if wordnet_result:
                            search_time = round((time.time() - start_time) * 1000, 2)
                            return {
                                'query': query,
                                'search_type': 'wordnet',
                                'word_count': 1,
                                'results': [{
                                    **wordnet_result,
                                    'search_type': 'wordnet',
                                    'source': 'wordnet'
                                }],
                                'search_time_ms': search_time,
                                'total_results': 1
                            }
                    except Exception as e:
                        print(f"WordNet search failed: {e}")

                # Nếu không có kết quả từ WordNet, thử fuzzy search
                fuzzy_results = self.fuzzy_dictionary_search(word)
                if fuzzy_results:
                    search_time = round((time.time() - start_time) * 1000, 2)
                    return {
                        'query': query,
                        'search_type': 'fuzzy_dictionary',
                        'word_count': 1,
                        'results': fuzzy_results,
                        'search_time_ms': search_time,
                        'total_results': len(fuzzy_results)
                    }
            
            for word in words:
                # Try exact lookup
                exact_result = self.exact_dictionary_lookup(word)
                if exact_result:
                    results.append(dict(exact_result, search_type='exact'))
                else:
                    # Fuzzy search for each word
                    fuzzy_results = self.fuzzy_dictionary_search(word, max_results=2)
                    results.extend(fuzzy_results)
            # Two words: try each word individually, then combine results
            
            # If no dictionary results, fallback to BM25
            if not results:
                bm25_results = self.bm25_search(query)
                search_time = round((time.time() - start_time) * 1000, 2)
                return {
                    'query': query,
                    'search_type': 'bm25_fallback',
                    'word_count': word_count,
                    'results': bm25_results,
                    'search_time_ms': search_time,
                    'total_results': len(bm25_results)
                }
            
            search_time = round((time.time() - start_time) * 1000, 2)
            return {
                'query': query,
                'search_type': 'dictionary_multi',
                'word_count': word_count,
                'results': results[:10],  # Limit results
                'search_time_ms': search_time,
                'total_results': len(results)
            }
        
        else:
            # More than 2 words: BM25 search
            bm25_results = self.bm25_search(query)
            search_time = round((time.time() - start_time) * 1000, 2)
            return {
                'query': query,
                'search_type': 'bm25',
                'word_count': word_count,
                'results': bm25_results,
                'search_time_ms': search_time,
                'total_results': len(bm25_results)
            }
    
    def get_suggestions(self, prefix, limit=5):
        """Get autocomplete suggestions"""
        if not prefix or len(prefix) < 2:
            return []
        
        prefix_lower = prefix.lower()
        suggestions = []
        
        # Dictionary suggestions
        dict_matches = [word for word in self.dictionary_data.keys() 
                       if word.startswith(prefix_lower)]
        suggestions.extend(sorted(dict_matches)[:limit//2])
        
        # Vocabulary suggestions
        vocab_matches = [word for word in self.vocabulary 
                        if word.startswith(prefix_lower) and word not in suggestions]
        suggestions.extend(sorted(vocab_matches)[:limit//2])
        
        return suggestions[:limit]

# Initialize search engine with WordNet check
search_engine = SmartSearchEngine()
search_engine.use_wordnet = initialize_nltk()

# Load data with logging
print("Loading dictionary and CSV data...")
search_engine.load_dictionary('dictionary.json')
search_engine.load_csv_documents(r'D:\TruyVan\project\backend\merged_file_unique.csv')
print("Initializing BM25 model...")
search_engine.initialize_bm25()
print("Search engine initialization complete")

@app.route('/search', methods=['GET'])
def search():
    """Main search endpoint"""
    query = request.args.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'No query provided', 'results': []})
    
    results = search_engine.smart_search(query)
    return jsonify(results)

@app.route('/suggest', methods=['GET'])
def suggest():
    """Autocomplete suggestions endpoint"""
    prefix = request.args.get('prefix', '').strip()
    limit = int(request.args.get('limit', 5))
    
    if not prefix:
        return jsonify([])
    
    suggestions = search_engine.get_suggestions(prefix, limit)
    return jsonify(suggestions)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'dictionary_entries': len(search_engine.dictionary_data),
        'csv_documents': len(search_engine.csv_documents),
        'bm25_ready': search_engine.bm25 is not None,
        'cached_files': {
            'bm25_cache': os.path.exists(search_engine.bm25_cache_file),
            'dictionary_cache': os.path.exists(search_engine.dictionary_cache_file)
        }
    })

@app.route('/rebuild-cache', methods=['POST'])
def rebuild_cache():
    """Force rebuild BM25 cache"""
    search_engine.initialize_bm25(force_rebuild=True)
    return jsonify({'status': 'cache rebuilt', 'message': 'BM25 model has been rebuilt and cached'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)