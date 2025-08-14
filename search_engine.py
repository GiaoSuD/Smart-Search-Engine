import csv
import re
import math
from collections import defaultdict, Counter
from rank_bm25 import BM25Okapi
import pickle
import os
import nltk
from nltk.corpus import wordnet as wn
import time

class BM25SearchEngine:
    """
    A BM25-based search engine for document retrieval
    """
    
    def __init__(self, k1=1.5, b=0.75):
        """
        Initialize the search engine
        
        Args:
            k1: Controls term frequency saturation point (typically 1.2-2.0)
            b: Controls how much document length normalizes (0-1, typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.preprocessed_docs = []
        self.inverted_index = defaultdict(list)
        self.vocabulary = set()
        self.bm25 = None
        self.avg_doc_length = 0
        self.doc_lengths = []
        self.cache_dir = 'cache'
        self.bm25_cache_file = os.path.join(self.cache_dir, 'bm25_model.pkl')
        self.csv_cache_file = os.path.join(self.cache_dir, 'csv_data.pkl')
        self.use_wordnet = True
        self.index_dir = 'index'
        self.index_file = os.path.join(self.index_dir, 'inverted_index.pkl')

    def preprocess_text(self, text):
        """
        Preprocess text for indexing and searching
        
        Args:
            text: Raw text string
            
        Returns:
            List of cleaned tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers (optional - you might want to keep them)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize by splitting on whitespace
        tokens = text.split()
        
        # Define stopwords (expand as needed)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
            'our', 'their', 'am', 'can', 'may', 'might', 'must', 'shall', 'up',
            'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
        }
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
        
        return tokens
    
    def build_inverted_index(self):
        """Build inverted index mapping words to document IDs"""
        self.inverted_index.clear()
        self.vocabulary.clear()
        
        for doc_id, tokens in enumerate(self.preprocessed_docs):
            # Add unique tokens to inverted index
            for token in set(tokens):
                self.inverted_index[token].append(doc_id)
                self.vocabulary.add(token)
    
    def save_to_cache(self):
        """Save BM25 model and processed documents to cache"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        cache_data = {
            'documents': self.documents,
            'preprocessed_docs': self.preprocessed_docs,
            'vocabulary': self.vocabulary,
            'inverted_index': self.inverted_index,
            'bm25': self.bm25,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length
        }
        
        try:
            with open(self.bm25_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Cache saved to {self.bm25_cache_file}")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def load_from_cache(self):
        """Load BM25 model and processed documents from cache"""
        try:
            if os.path.exists(self.bm25_cache_file):
                with open(self.bm25_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                self.documents = cache_data['documents']
                self.preprocessed_docs = cache_data['preprocessed_docs']
                self.vocabulary = cache_data['vocabulary']
                self.inverted_index = cache_data['inverted_index']
                self.bm25 = cache_data['bm25']
                self.doc_lengths = cache_data['doc_lengths']
                self.avg_doc_length = cache_data['avg_doc_length']
                
                print(f"Loaded cache from {self.bm25_cache_file}")
                print(f"Documents: {len(self.documents)}")
                print(f"Vocabulary size: {len(self.vocabulary)}")
                return True
        except Exception as e:
            print(f"Error loading cache: {e}")
        return False

    def load_csv_data(self, csv_file, force_rebuild=False):
        """Load data from CSV file with caching"""
        if not force_rebuild and self.load_from_cache():
            return
            
        print(f"Loading CSV data from {csv_file}...")
        self.documents.clear()
        self.preprocessed_docs.clear()
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    doc = {
                        'id': len(self.documents),
                        'original_id': row.get('ID', str(len(self.documents))),
                        'title': row.get('Title', ''),
                        'content': row.get('Content', '')
                    }
                    self.documents.append(doc)
                    
                    # Preprocess combined title and content
                    full_text = f"{doc['title']} {doc['content']}"
                    tokens = self.preprocess_text(full_text)
                    self.preprocessed_docs.append(tokens)
                    
        except FileNotFoundError:
            print(f"File {csv_file} not found. Creating sample data...")
            self.create_sample_data()
        
        # Calculate document lengths
        self.doc_lengths = [len(doc) for doc in self.preprocessed_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Build indexes
        self.build_inverted_index()
        
        # Initialize BM25
        if self.preprocessed_docs:
            self.bm25 = BM25Okapi(self.preprocessed_docs)
        
        print(f"Search engine initialized with {len(self.documents)} documents")
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        sample_data = [
            {
                'original_id': '1',
                'title': 'Python Programming Language',
                'content': 'Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.'
            },
            {
                'original_id': '2',
                'title': 'Machine Learning Fundamentals',
                'content': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build mathematical models based on training data in order to make predictions or decisions without being explicitly programmed to do so.'
            },
            {
                'original_id': '3',
                'title': 'Web Development Technologies',
                'content': 'Web development refers to the tasks associated with developing websites for hosting via intranet or internet. The web development process includes web design, web content development, client-side/server-side scripting, and network security configuration. Modern web development involves HTML5, CSS3, JavaScript frameworks like React and Vue, backend technologies like Node.js and Python, and databases.'
            },
            {
                'original_id': '4',
                'title': 'Data Science and Analytics',
                'content': 'Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. Data science combines aspects of statistics, computer science, information science, and domain expertise to analyze and interpret complex data for decision-making.'
            },
            {
                'original_id': '5',
                'title': 'Artificial Intelligence Research',
                'content': 'Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.'
            },
            {
                'original_id': '6',
                'title': 'Database Management Systems',
                'content': 'A database management system is system software for creating and managing databases. DBMS provides users and programmers with a systematic way to create, retrieve, update and manage data. Examples include MySQL, PostgreSQL, MongoDB, and SQLite. Modern databases support ACID properties and can handle big data workloads.'
            },
            {
                'original_id': '7',
                'title': 'Cloud Computing Platforms',
                'content': 'Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations from central servers. Major cloud providers include Amazon Web Services, Microsoft Azure, and Google Cloud Platform.'
            },
            {
                'original_id': '8',
                'title': 'Cybersecurity Best Practices',
                'content': 'Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. These attacks are usually aimed at accessing, changing, or destroying sensitive information, extorting money from users, or interrupting normal business processes. Effective cybersecurity measures include firewalls, encryption, multi-factor authentication, and regular security audits.'
            },
            {
                'original_id': '9',
                'title': 'Mobile Application Development',
                'content': 'Mobile app development is the act or process by which a mobile app is developed for mobile devices, such as personal digital assistants, enterprise digital assistants or mobile phones. These applications can be pre-installed on phones during manufacturing platforms, or delivered as web applications using server-side or client-side processing.'
            },
            {
                'original_id': '10',
                'title': 'DevOps and Continuous Integration',
                'content': 'DevOps is a set of practices that combines software development and IT operations. It aims to shorten the systems development life cycle and provide continuous delivery with high software quality. DevOps tools include version control systems like Git, CI/CD pipelines, containerization with Docker, and orchestration with Kubernetes.'
            }
        ]
        
        for i, data in enumerate(sample_data):
            doc = {
                'id': i,
                'original_id': data['original_id'],
                'title': data['title'],
                'content': data['content']
            }
            self.documents.append(doc)
            
            # Preprocess
            full_text = f"{doc['title']} {doc['content']}"
            tokens = self.preprocess_text(full_text)
            self.preprocessed_docs.append(tokens)
    
    def search_documents(self, query, limit=10):
        """
        Search documents using BM25 ranking
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of ranked documents with scores
        """
        if not query.strip() or not self.bm25:
            return []
        
        # Preprocess query
        query_tokens = self.preprocess_text(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Create list of (document_id, score) pairs
        scored_docs = [(i, scores[i]) for i in range(len(scores))]
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for doc_id, score in scored_docs[:limit]:
            if score > 0:  # Only include documents with positive relevance
                doc = self.documents[doc_id]
                snippet = self.generate_snippet(doc['content'], query_tokens)
                
                result = {
                    'id': doc['original_id'],
                    'title': doc['title'],
                    'snippet': snippet,
                    'full_content': doc['content'],
                    'score': round(score, 4)
                }
                results.append(result)
        
        return results
    
    def generate_snippet(self, content, query_tokens, snippet_length=200):
        """
        Generate a snippet that highlights query terms
        
        Args:
            content: Full document content
            query_tokens: List of query tokens
            snippet_length: Maximum snippet length
            
        Returns:
            Snippet string
        """
        if not content:
            return ""
        
        content_lower = content.lower()
        
        # Find the best position to start the snippet
        best_pos = 0
        for token in query_tokens:
            pos = content_lower.find(token.lower())
            if pos != -1:
                # Start snippet before the match for context
                best_pos = max(0, pos - 30)
                break
        
        # Extract snippet
        snippet = content[best_pos:best_pos + snippet_length]
        
        # Clean up snippet boundaries
        if best_pos > 0:
            # Find a good place to start (after a space)
            space_pos = snippet.find(' ')
            if space_pos != -1:
                snippet = snippet[space_pos + 1:]
            snippet = "..." + snippet
        
        if len(content) > best_pos + snippet_length:
            # Find a good place to end (before a space)
            last_space = snippet.rfind(' ')
            if last_space != -1:
                snippet = snippet[:last_space]
            snippet = snippet + "..."
        
        return snippet.strip()
    
    def get_suggestions(self, prefix, limit=5):
        """
        Get autocomplete suggestions for a prefix
        
        Args:
            prefix: Text prefix to match
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested words
        """
        if not prefix or len(prefix) < 2:
            return []
        
        prefix_lower = prefix.lower()
        
        # Find words that start with the prefix
        matching_words = []
        for word in self.vocabulary:
            if word.startswith(prefix_lower):
                matching_words.append(word)
        
        # Sort alphabetically and limit results
        matching_words.sort()
        return matching_words[:limit]
    
    def get_stats(self):
        """Get search engine statistics"""
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'average_document_length': round(self.avg_doc_length, 2),
            'total_tokens': sum(self.doc_lengths)
        }
    
    def initialize_nltk(self):
        """Initialize NLTK WordNet"""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            print("NLTK WordNet initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing NLTK WordNet: {e}")
            return False
            
    def get_wordnet_info(self, word):
        """Get word information from WordNet"""
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
                word_info['definitions'].append({
                    'definition': synset.definition(),
                    'pos': synset.pos()
                })
                word_info['examples'].extend(synset.examples())
                
                for lemma in synset.lemmas():
                    word_info['synonyms'].add(lemma.name())
                    if lemma.antonyms():
                        word_info['antonyms'].add(lemma.antonyms()[0].name())
                
                word_info['pos_tags'].add(synset.pos())
            
            word_info['synonyms'] = list(word_info['synonyms'])
            word_info['antonyms'] = list(word_info['antonyms'])
            word_info['pos_tags'] = list(word_info['pos_tags'])
            
            return word_info
            
        except Exception as e:
            print(f"WordNet lookup error for word '{word}': {e}")
            return None

    def smart_search(self, query, limit=10):
        """Combined search using WordNet and BM25"""
        query = query.strip()
        if not query:
            return {'error': 'Empty query', 'results': []}

        words = query.split()
        word_count = len(words)
        start_time = time.time()

        # 1-2 từ: Dùng WordNet
        if word_count <= 2:
            results = []
            
            # Xử lý stopword nếu là 2 từ
            if word_count == 2:
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
                if words[0].lower() in stopwords:
                    word = words[1]
                    wordnet_result = self.get_wordnet_info(word)
                    if wordnet_result:
                        search_time = round((time.time() - start_time) * 1000, 2)
                        return {
                            'query': query,
                            'search_type': 'wordnet',
                            'results': [wordnet_result],
                            'search_time_ms': search_time
                        }
            
            # Tìm kiếm WordNet cho từng từ
            for word in words:
                wordnet_result = self.get_wordnet_info(word)
                if wordnet_result:
                    results.append(wordnet_result)
            
            if results:
                search_time = round((time.time() - start_time) * 1000, 2)
                return {
                    'query': query,
                    'search_type': 'wordnet',
                    'results': results,
                    'search_time_ms': search_time
                }

        # 3 từ trở lên: Chỉ dùng BM25 search trên file CSV
        else:
            # Check if BM25 model exists
            if not self.bm25:
                return {
                    'query': query,
                    'search_type': 'error',
                    'error': 'BM25 model not initialized',
                    'results': []
                }

            # Search using BM25
            bm25_results = self.search_documents(query, limit)
            
            if not bm25_results:
                return {
                    'query': query,
                    'search_type': 'bm25',
                    'word_count': word_count,
                    'results': [],
                    'search_time_ms': round((time.time() - start_time) * 1000, 2),
                    'total_results': 0,
                    'message': 'No matching documents found'
                }

            search_time = round((time.time() - start_time) * 1000, 2)
            return {
                'query': query,
                'search_type': 'bm25',
                'word_count': word_count,
                'results': bm25_results,
                'search_time_ms': search_time,
                'total_results': len(bm25_results)
            }
    
    def load_index(self, index_file=None):
        """Load existing search index"""

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
    
        if index_file is None:
            index_file = os.path.join(DATA_DIR, 'search_index.pkl')

        try:
            print(f"Loading search index from {index_file}...")
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25 = index_data['bm25']
            self.documents = index_data['documents']
            self.preprocessed_docs = index_data['preprocessed_docs']
            self.vocabulary = index_data['vocabulary']
        
            print(f"Loaded index with {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
