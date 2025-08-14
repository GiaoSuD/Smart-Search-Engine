from rank_bm25 import BM25Okapi
import csv
import pickle
import os
import re
from datetime import datetime
csv.field_size_limit(2**20)

def preprocess_text(text):
    """Clean and tokenize text"""
    if not text:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and short tokens
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    return tokens

def create_index(csv_file, output_dir='index'):
    """Create BM25 index from CSV file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Creating index from {csv_file}...")
    
    documents = []
    preprocessed_docs = []
    vocabulary = set()
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                doc = {
                    'id': len(documents),
                    'original_id': row.get('ID', str(len(documents))),
                    'title': row.get('Title', ''),
                    'content': row.get('Content', ''),
                    'source': 'csv'
                }
                documents.append(doc)
                
                # Preprocess text
                full_text = f"{doc['title']} {doc['content']}"
                tokens = preprocess_text(full_text)
                preprocessed_docs.append(tokens)
                vocabulary.update(tokens)
                
                # Show progress
                if len(documents) % 1000 == 0:
                    print(f"Processed {len(documents)} documents...")
        
        # Create BM25 model
        print("Building BM25 model...")
        bm25 = BM25Okapi(preprocessed_docs)
        
        # Save index files
        index_data = {
            'bm25': bm25,
            'documents': documents,
            'preprocessed_docs': preprocessed_docs,
            'vocabulary': vocabulary,
            'created_at': datetime.now().isoformat()
        }
        
        output_file = os.path.join(output_dir, 'search_index.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(index_data, f)
            
        print(f"\nIndex creation complete:")
        print(f"- Documents indexed: {len(documents)}")
        print(f"- Vocabulary size: {len(vocabulary)}")
        print(f"- Index saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    csv_file = r"D:\TruyVan\data\Simple_wiki_data.csv"  # Replace with your CSV file path
    create_index(csv_file)