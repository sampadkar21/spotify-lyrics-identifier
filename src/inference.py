import weaviate
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pandas as pd
from tqdm.auto import tqdm

class SongIdentifier:
    def __init__(
        self, 
        db_path: str = '/kaggle/working/vectordb',
        collection_name: str = 'Spotify_songs',
        model_name: str = 'ibm-granite/granite-embedding-small-english-r2'
    ):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(model_name)
        
        print("Connecting to vector database...")
        self.client = weaviate.connect_to_embedded(
            version="latest",
            persistence_data_path=db_path,
            environment_variables={"LOG_LEVEL": "error"}
        )
        
        self.collection = self.client.collections.get(collection_name)
        print("Song Identifier ready!")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        alpha: float = 0.7,
        return_full_text: bool = False
    ) -> List[Dict]:
        query = query.lower().strip()
        query_vector = self.embedder.encode(query).tolist()
        
        response = self.collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=top_k
        )
        
        results = []
        for rank, obj in enumerate(response.objects, 1):
            text = obj.properties['text']
            if not return_full_text:
                text = text[:150] + '...' if len(text) > 150 else text
                
            results.append({
                'rank': rank,
                'song': obj.properties['song'].title(),
                'artist': obj.properties['artist'].title(),
                'text_preview': text,
                'similarity_score': getattr(obj.metadata, 'score', 0.0)
            })
        
        return results
    
    def calculate_accuracy(
        self,
        test_df: pd.DataFrame,
        k_values: List[int] = [1, 3, 5],
        alpha: float = 0.7
    ) -> Dict[int, float]:
        max_k = max(k_values)
        correct_matches = {k: 0 for k in k_values}
        total_samples = len(test_df)
        
        print(f"Testing on {total_samples} samples...")
        
        for _, row in tqdm(test_df.iterrows(), total=total_samples):
            query_text = row['text'].lower()
            target_song = row['song'].lower()
            target_artist = row['artist'].lower()
            
            query_vector = self.embedder.encode(query_text).tolist()
            
            response = self.collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=max_k
            )
            
            found_rank = -1
            for rank, obj in enumerate(response.objects, 1):
                if (obj.properties['song'] == target_song and 
                    obj.properties['artist'] == target_artist):
                    found_rank = rank
                    break
            
            if found_rank != -1:
                for k in k_values:
                    if found_rank <= k:
                        correct_matches[k] += 1
        
        results = {}
        for k in k_values:
            accuracy = (correct_matches[k] / total_samples) * 100
            results[k] = round(accuracy, 2)
        
        return results
    
    def close(self):
        if self.client:
            self.client.close()
            print("Connection closed")

def demo_usage():
    identifier = SongIdentifier()
    test_query = "sample lyrics about love and heartbreak"
    results = identifier.search(test_query, top_k=3)
    
    print("\nSearch Results:")
    for result in results:
        print(f"\n{result['rank']}. {result['song']} - {result['artist']}")
        print(f"Similarity: {result['similarity_score']:.2%}")
    
    identifier.close()

if __name__ == "__main__":
    demo_usage()
