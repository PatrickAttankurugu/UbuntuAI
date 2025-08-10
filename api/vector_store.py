import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json
import os
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

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
        
        # Initialize embedding service
        self.embedding_service = None
        self._initialize_embedding_service()
    
    def _initialize_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "African Business Knowledge Base - Gemini Embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _initialize_embedding_service(self):
        """Initialize the Gemini embedding service"""
        try:
            from utils.embeddings import embedding_service
            if embedding_service is not None:
                self.embedding_service = embedding_service
                logger.info("Gemini embedding service connected to vector store")
            else:
                logger.error("Embedding service not available")
        except ImportError as e:
            logger.error(f"Failed to import embedding service: {e}")
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str] = None) -> bool:
        try:
            if not self.embedding_service:
                logger.error("Embedding service not available")
                return False
                
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Create embeddings using Gemini
            logger.info(f"Creating embeddings for {len(documents)} documents...")
            embeddings = self.embedding_service.create_embeddings_batch(
                documents, 
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            valid_docs = []
            valid_embeddings = []
            valid_metadatas = []
            valid_ids = []
            
            # Filter out failed embeddings
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_docs.append(documents[i])
                    valid_embeddings.append(embedding)
                    valid_metadatas.append(metadatas[i])
                    valid_ids.append(ids[i])
                else:
                    logger.warning(f"Failed to create embedding for document {i}")
            
            if valid_docs:
                # Add to ChromaDB
                self.collection.add(
                    documents=valid_docs,
                    embeddings=valid_embeddings,
                    metadatas=valid_metadatas,
                    ids=valid_ids
                )
                logger.info(f"Successfully added {len(valid_docs)} documents to vector store")
            else:
                logger.error("No valid embeddings created")
                return False
            
            success_rate = len(valid_docs) / len(documents)
            logger.info(f"Embedding success rate: {success_rate:.2%}")
            
            return success_rate > 0.5  # Consider successful if > 50% embeddings created
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search(self, 
              query: str, 
              n_results: int = 10,
              filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            if not self.embedding_service:
                logger.error("Embedding service not available for search")
                return []
            
            # Create query embedding using Gemini
            query_embedding = self.embedding_service.create_query_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to create query embedding")
                return []
            
            # Prepare ChromaDB filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    # Convert distance to similarity (ChromaDB returns cosine distance)
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": max(0.0, min(1.0, similarity)),  # Clamp to [0,1]
                        "distance": distance
                    })
            
            logger.debug(f"Search returned {len(formatted_results)} results for query: {query[:50]}...")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def search_by_embedding(self,
                           query_embedding: List[float],
                           n_results: int = 10,
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search using a pre-computed embedding"""
        try:
            # Prepare ChromaDB filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1 - distance
                    
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "similarity": max(0.0, min(1.0, similarity)),
                        "distance": distance
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching by embedding: {e}")
            return []
    
    def update_document(self, doc_id: str, document: str = None, metadata: Dict[str, Any] = None) -> bool:
        try:
            if not self.embedding_service:
                logger.error("Embedding service not available for update")
                return False
                
            update_data = {"ids": [doc_id]}
            
            if document:
                embedding = self.embedding_service.create_document_embedding(document)
                if embedding:
                    update_data["documents"] = [document]
                    update_data["embeddings"] = [embedding]
                else:
                    logger.error(f"Failed to create embedding for document update: {doc_id}")
                    return False
            
            if metadata:
                update_data["metadatas"] = [metadata]
            
            self.collection.update(**update_data)
            logger.info(f"Successfully updated document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents(self, ids: List[str]) -> bool:
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(10, count)
            sample_results = None
            
            if count > 0:
                try:
                    sample_results = self.collection.get(
                        limit=sample_size,
                        include=["metadatas"]
                    )
                except Exception as e:
                    logger.warning(f"Could not get sample data: {e}")
            
            # Analyze metadata if available
            metadata_analysis = {}
            if sample_results and sample_results.get("metadatas"):
                countries = set()
                sectors = set()
                types = set()
                
                for metadata in sample_results["metadatas"]:
                    if metadata:
                        if metadata.get("country"):
                            countries.add(metadata["country"])
                        if metadata.get("sector"):
                            if isinstance(metadata["sector"], list):
                                sectors.update(metadata["sector"])
                            else:
                                sectors.add(metadata["sector"])
                        if metadata.get("type"):
                            types.add(metadata["type"])
                
                metadata_analysis = {
                    "unique_countries": len(countries),
                    "unique_sectors": len(sectors),
                    "unique_types": len(types),
                    "sample_countries": list(countries)[:5],
                    "sample_sectors": list(sectors)[:5],
                    "sample_types": list(types)[:5]
                }
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_service": "Gemini" if self.embedding_service else "None",
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimensions": settings.EMBEDDING_DIMENSIONS,
                "metadata_analysis": metadata_analysis
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def backup_collection(self, backup_path: str) -> bool:
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
            
            backup_data = {
                "documents": all_data.get("documents", []),
                "metadatas": all_data.get("metadatas", []),
                "ids": all_data.get("ids", []),
                "collection_name": self.collection_name,
                "embedding_model": settings.EMBEDDING_MODEL,
                "backup_timestamp": str(datetime.now())
            }
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"Successfully backed up {len(backup_data['documents'])} documents to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up collection: {e}")
            return False
    
    def restore_collection(self, backup_path: str) -> bool:
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            # Delete existing collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except ValueError:
                pass  # Collection might not exist
            
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "African Business Knowledge Base - Restored from backup"}
            )
            
            # Restore documents (this will regenerate embeddings)
            success = self.add_documents(
                documents=backup_data["documents"],
                metadatas=backup_data["metadatas"],
                ids=backup_data["ids"]
            )
            
            if success:
                logger.info(f"Successfully restored {len(backup_data['documents'])} documents from backup")
            else:
                logger.warning("Restore completed with some errors")
                
            return success
            
        except Exception as e:
            logger.error(f"Error restoring collection: {e}")
            return False
    
    def similarity_search_with_score(self, 
                                   query: str,
                                   k: int = 10,
                                   filter: Dict[str, Any] = None) -> List[tuple]:
        """
        Perform similarity search and return results with scores
        Compatible with LangChain interface
        """
        results = self.search(query, n_results=k, filters=filter)
        
        # Convert to LangChain-compatible format
        documents_with_scores = []
        for result in results:
            # Create a simple document object
            doc = type('Document', (), {
                'page_content': result['content'],
                'metadata': result['metadata']
            })()
            
            score = result['similarity']
            documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def add_texts(self,
                 texts: List[str],
                 metadatas: List[Dict[str, Any]] = None,
                 ids: List[str] = None) -> List[str]:
        """
        Add texts to the vector store
        Compatible with LangChain interface
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if ids is None:
            ids = [f"text_{i}_{hash(text) % 10000}" for i, text in enumerate(texts)]
        
        success = self.add_documents(texts, metadatas, ids)
        
        if success:
            return ids
        else:
            return []

# Global vector store instance
try:
    vector_store = ChromaVectorStore()
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vector_store = None