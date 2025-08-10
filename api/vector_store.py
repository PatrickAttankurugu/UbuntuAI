import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import os
from config.settings import settings
from utils.embeddings import embedding_service

class ChromaVectorStore:
    def __init__(self):
        self.persist_directory = settings.CHROMA_PERSIST_DIRECTORY
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = settings.COLLECTION_NAME
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "African Business Knowledge Base"}
            )
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str] = None) -> bool:
        try:
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            embeddings = embedding_service.create_embeddings_batch(documents)
            
            valid_docs = []
            valid_embeddings = []
            valid_metadatas = []
            valid_ids = []
            
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_docs.append(documents[i])
                    valid_embeddings.append(embedding)
                    valid_metadatas.append(metadatas[i])
                    valid_ids.append(ids[i])
            
            if valid_docs:
                self.collection.add(
                    documents=valid_docs,
                    embeddings=valid_embeddings,
                    metadatas=valid_metadatas,
                    ids=valid_ids
                )
            
            return len(valid_docs) == len(documents)
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def search(self, 
              query: str, 
              n_results: int = 10,
              filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            query_embedding = embedding_service.create_embedding(query)
            
            if query_embedding is None:
                return []
            
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": 1 - results["distances"][0][i] if results["distances"] else 0.0,
                        "distance": results["distances"][0][i] if results["distances"] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def update_document(self, doc_id: str, document: str = None, metadata: Dict[str, Any] = None) -> bool:
        try:
            update_data = {"ids": [doc_id]}
            
            if document:
                embedding = embedding_service.create_embedding(document)
                if embedding:
                    update_data["documents"] = [document]
                    update_data["embeddings"] = [embedding]
            
            if metadata:
                update_data["metadatas"] = [metadata]
            
            self.collection.update(**update_data)
            return True
            
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}
    
    def backup_collection(self, backup_path: str) -> bool:
        try:
            all_data = self.collection.get()
            
            backup_data = {
                "documents": all_data["documents"],
                "metadatas": all_data["metadatas"],
                "ids": all_data["ids"]
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error backing up collection: {e}")
            return False
    
    def restore_collection(self, backup_path: str) -> bool:
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            self.client.delete_collection(name=self.collection_name)
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "African Business Knowledge Base - Restored"}
            )
            
            return self.add_documents(
                documents=backup_data["documents"],
                metadatas=backup_data["metadatas"],
                ids=backup_data["ids"]
            )
            
        except Exception as e:
            print(f"Error restoring collection: {e}")
            return False

vector_store = ChromaVectorStore()