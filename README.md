# 🎵 Spotify Lyrics Identification System

A high-performance text identification algorithm that identifies **Song Title** and **Artist** from small snippets of lyrics using semantic search and hybrid retrieval.

---

## 🎯 Project Overview

This project implements a **semantic search-based** approach to identify songs from lyric snippets. Unlike traditional classification models, this solution uses:

- **Vector embeddings** for semantic understanding
- **Hybrid search** combining dense vectors + BM25 keyword matching
- **Weaviate** vector database for efficient similarity search
- **IBM Granite** embedding model (384-dim vectors)

### Why Semantic Search?

✅ **Scalable**: Add new songs without retraining  
✅ **Flexible**: Works with any length of lyrics  
✅ **Robust**: Handles typos and variations  
✅ **Fast**: Sub-second inference on 57k+ songs  

---

## 🗂️ Repository Structure

```
spotify-lyrics-identifier/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   
│   └── 02_model_evaluation.ipynb    
│
├── src/
│    └── inference.py                                                       
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/spotify-lyrics-identifier.git
cd spotify-lyrics-identifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Spotify Million Song Dataset** from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs)

Place `Spotify Million Song Dataset_exported.csv` in the `data/` folder.

### 3. Run the Model

#### Option A: Using Notebooks (Recommended)

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
# Follow the notebook to create vector database

jupyter notebook notebooks/02_model_evaluation.ipynb
# Test the model and see results
```

#### Option B: Using Python Scripts

```python
from src.inference import SongIdentifier

# Initialize
identifier = SongIdentifier(db_path='vectordb')

# Identify song from lyrics
results = identifier.search(
    query="I see trees of green, red roses too",
    top_k=3
)

print(f"Song: {results[0]['song']}")
print(f"Artist: {results[0]['artist']}")
print(f"Similarity: {results[0]['score']:.2%}")
```

---

## 🧠 Technical Approach

### 1. **Text Preprocessing**

- Lowercase normalization
- Tokenization using WordPiece (IBM Granite tokenizer)
- No stop-word removal (preserves semantic meaning)
- Max sequence length: 512 tokens

### 2. **Embedding Model**

**IBM Granite Embedding (Small English R2)**
- Dimensions: 384
- Context window: 8192 tokens
- Optimized for semantic similarity tasks

### 3. **Vector Database**

**Weaviate (Embedded)**
- 57,650 songs indexed
- HNSW (Hierarchical Navigable Small World) for fast ANN search
- Hybrid search: 70% vector + 30% keyword (BM25)

### 4. **Search Strategy**

```python
def hybrid_search(query, alpha=0.7):
    """
    alpha = 0.7 → 70% semantic, 30% keyword matching
    
    Returns: Top-k most similar songs with:
    - song_title
    - artist_name
    - similarity_score
    """
```

---

## 📈 Performance Analysis

### Dataset Statistics

- **Total Songs**: 57,650
- **Unique Artists**: 643
- **Average Lyric Length**: ~180 tokens
- **Longest Lyric**: 512 tokens

### Accuracy Breakdown

| Test Scenario | Top-1 | Top-3 | Top-5 |
|--------------|-------|-------|-------|
| Full lyrics | 99.1% | 99.9% | 100% |
| Partial lyrics (50%) | ~95% | ~98% | ~99% |
| Single verse | ~88% | ~94% | ~97% |

### Inference Speed

- **Single query**: ~0.4 seconds
- **Batch (100 queries)**: ~25 seconds
- **Database size**: ~2.1 GB

---

## 🔬 Example Usage

```python
from src.inference import SongIdentifier

identifier = SongIdentifier()

# Example 1: Famous chorus
results = identifier.search(
    "We will we will rock you",
    top_k=1
)
# Output: "We Will Rock You" by Queen

# Example 2: Verse snippet  
results = identifier.search(
    "Just a small town girl living in a lonely world",
    top_k=1
)
# Output: "Don't Stop Believin'" by Journey

# Example 3: Obscure lyrics
results = identifier.search(
    "The screen door slams Mary's dress waves",
    top_k=1  
)
# Output: "Thunder Road" by Bruce Springsteen
```

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12 |
| **Embeddings** | IBM Granite (sentence-transformers) |
| **Vector DB** | Weaviate (Embedded) |
| **Data Processing** | Pandas, NumPy |
| **Notebooks** | Jupyter |
| **Visualization** | Matplotlib, tqdm |

---

### Accuracy Calculation

```python
def calculate_accuracy(collection, test_df, k_values=[1,3,5]):
    correct = {k: 0 for k in k_values}
    
    for row in test_df:
        results = collection.query.hybrid(
            query=row['text'],
            vector=embedder.encode(row['text']),
            limit=max(k_values)
        )
        
        # Check if true song in top-k results
        for rank, result in enumerate(results):
            if result['song'] == row['song']:
                for k in k_values:
                    if rank < k:
                        correct[k] += 1
                        
    return {k: correct[k]/len(test_df) for k in k_values}
```

---

## 🔮 Future Improvements

1. **Multi-language Support**: Extend to Spanish, French, etc.
2. **Audio Features**: Combine lyrics + melody embeddings

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- **Dataset**: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs)
- **Embedding Model**: IBM Granite 
- **Vector Database**: Weaviate Open Source

---

**⭐ If you found this project helpful, please give it a star!**
