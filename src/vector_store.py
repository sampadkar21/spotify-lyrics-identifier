import weaviate
import weaviate.classes.config as wc
from tqdm.auto import tqdm
from typing import List, Dict, Optional

class VectorStore:
    def __init__(
        self, 
        db_name: str, 
        persistent_path: str, 
        binary_path: str, 
        embedding_model
    ):
        self.db_name = db_name
        self.embedding_model = embedding_model
        self.persistent_path = persistent_path
        self.binary_path = binary_path
        self.client = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            self.client = weaviate.connect_to_embedded(
                version="latest",
                binary_path=self.binary_path,
                persistence_data_path=self.persistent_path,
                environment_variables={"LOG_LEVEL": "error"} 
            )
            print(f"Vector Store initialized successfully!")
        except Exception as e:
            print(f"Vector Store initialization failed: {e}")

    def create_store(self):
        if not self.client:
            return

        if self.client.collections.exists(self.db_name):
            self.client.collections.delete(self.db_name)
            print(f"Dropped existing collection: '{self.db_name}'")

        self.client.collections.create(
            name=self.db_name,
            description="Spotify Songs Collection",
            vectorizer_config=[
                wc.Configure.NamedVectors.none(name="default")
            ],
            properties=[
                wc.Property(name="song", data_type=wc.DataType.TEXT),
                wc.Property(name="artist", data_type=wc.DataType.TEXT),
                wc.Property(name="text", data_type=wc.DataType.TEXT),
            ]
        )
        print(f"Collection '{self.db_name}' created successfully.")

    def add_documents(self, data_list: List[Dict]):
        if not self.client:
            return

        collection = self.client.collections.get(self.db_name)

        try:
            with collection.batch.dynamic() as batch:
                for doc in tqdm(data_list, desc="Processing"):
                    vector = self.embedding_model.encode(
                        doc["text"], 
                        show_progress_bar=False
                    ).tolist()
                    
                    batch.add_object(
                        properties={
                            "song": doc["song"],
                            "artist": doc["artist"],
                            "text": doc["text"]
                        },
                        vector={"default": vector}
                    )

            if len(collection.batch.failed_objects) > 0:
                print(f"Failed to import {len(collection.batch.failed_objects)} objects")
                print(collection.batch.failed_objects[0].message)
            else:
                print(f"Successfully added {len(data_list)} documents")
                
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        alpha: float = 0.7
    ) -> List[Dict]:
        if not self.client:
            return []
        
        collection = self.client.collections.get(self.db_name)
        
        query_vector = self.embedding_model.encode(query).tolist()
        
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=top_k
        )
        
        results = []
        for obj in response.objects:
            results.append({
                'song': obj.properties['song'],
                'artist': obj.properties['artist'],
                'text': obj.properties['text'][:200] + '...',
                'score': getattr(obj.metadata, 'score', 0.0)
            })
        
        return results
    
    def close(self):
        if self.client:
            self.client.close()
            print("Database connection closed")
