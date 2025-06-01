"""
ÉMERGENCE V4 - RAG Manager COMPLET CORRIGÉ
Compatible 100% Database V4 + Backend FastAPI
TOUTES les méthodes requises par le backend !
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# Import SEULEMENT database (validé)
from .database import get_database

logger = logging.getLogger(__name__)

class RAGManagerV4Minimal:
    """
    RAG Manager V4 COMPLET avec TOUTES les méthodes backend
    Compatible 100% Database V4 + FastAPI main.py
    """
    
    def __init__(self):
        self.db = get_database()
        self.initialized = False
        
        try:
            # Test connexion database
            health = self.db.health_check()
            if health['status'] in ['healthy', 'degraded']:
                self.initialized = True
                logger.info(f"✅ RAG Manager V4 COMPLET initialisé - DB: {health['status']}")
            else:
                logger.error(f"❌ Database non disponible: {health}")
        except Exception as e:
            logger.error(f"❌ Erreur init RAG: {e}")
    
    def add_document(self, filename: str, file_path: str, content: str = None, 
                    chunks: List[str] = None) -> Tuple[bool, str, Dict]:
        """
        Upload document - Compatible Database V4 API
        """
        if not self.initialized:
            return False, "RAG Manager non initialisé", {}
        
        start_time = time.time()
        
        try:
            # 1. Hash + taille
            if content:
                file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                file_size = len(content.encode('utf-8'))
            else:
                if not Path(file_path).exists():
                    return False, f"Fichier non trouvé: {file_path}", {}
                
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                    file_hash = hashlib.sha256(file_content).hexdigest()
                    file_size = len(file_content)
            
            logger.info(f"📄 Upload: {filename} ({file_size} bytes)")
            
            # 2. ✅ Database V4: add_document SANS metadata
            try:
                doc_id = self.db.add_document(
                    filename=filename,
                    file_path=file_path, 
                    file_hash=file_hash,
                    file_size=file_size
                )
                logger.info(f"✅ Document en base: {doc_id}")
            except Exception as e:
                logger.error(f"❌ add_document failed: {e}")
                return False, f"Erreur DB: {e}", {}
            
            # 3. Chunking simple
            if not chunks and content:
                chunks = self._simple_chunk_content(content)
            elif not chunks:
                chunks = []
            
            # 4. ✅ Database V4: add_chunks format correct
            chunk_count = 0
            if chunks:
                chunks_data = []
                for i, chunk_content in enumerate(chunks):
                    chunks_data.append({
                        'content': chunk_content,
                        'metadata': {
                            'chunk_index': i,
                            'source_file': filename,
                            'created_at': datetime.now().isoformat()
                        }
                    })
                
                try:
                    chunk_count = self.db.add_chunks(doc_id, chunks_data)
                    logger.info(f"✅ {chunk_count} chunks ajoutés")
                except Exception as e:
                    logger.error(f"❌ add_chunks failed: {e}")
            
            processing_time = time.time() - start_time
            
            result = {
                'document_id': doc_id,
                'filename': filename,
                'chunks_added': chunk_count,
                'vectors_added': 0,
                'processing_time': processing_time,
                'status': 'success'
            }
            
            logger.info(f"🎯 Upload réussi: {filename} - {chunk_count} chunks en {processing_time:.2f}s")
            return True, "Document ajouté avec succès", result
            
        except Exception as e:
            logger.error(f"❌ Erreur upload {filename}: {e}")
            return False, f"Erreur: {e}", {}
    
    def _simple_chunk_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Chunking simple par taille"""
        if not content or len(content) <= chunk_size:
            return [content] if content else []
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            
            if end < len(content):
                space_pos = content.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
            if start >= len(content):
                break
        
        return chunks
    
    def search(self, query: str, limit: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Recherche FTS5 SQLite
        """
        if not self.initialized:
            return [], {"error": "RAG non initialisé"}
        
        start_time = time.time()
        
        try:
            # ✅ Database V4: search_chunks
            results = self.db.search_chunks(query, limit=limit)
            
            search_results = []
            for result in results:
                search_results.append({
                    'id': result.get('id'),
                    'content': result.get('content', ''),
                    'filename': result.get('filename', 'unknown'),
                    'document_id': result.get('document_id'),
                    'chunk_index': result.get('chunk_index', 0),
                    'metadata': result.get('metadata', {}),
                    'search_source': result.get('source', 'fts5'),
                    'relevance_score': 0.8
                })
            
            search_stats = {
                'query': query,
                'total_results': len(search_results),
                'processing_time': time.time() - start_time,
                'search_type': 'fts5_only'
            }
            
            logger.info(f"🔍 Recherche: {len(search_results)} résultats pour '{query}'")
            return search_results, search_stats
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return [], {"error": str(e)}
    
    def get_context_for_agent(self, query: str, agent_name: str = "Agent", 
                             max_results: int = 5) -> str:
        """
        Contexte pour agents
        """
        try:
            search_results, stats = self.search(query, limit=max_results)
            
            if not search_results:
                return f"CONTEXTE RAG: Aucun document trouvé pour '{query[:50]}...'"
            
            context_parts = [
                f"CONTEXTE RAG ({len(search_results)} documents trouvés):"
            ]
            
            for i, result in enumerate(search_results, 1):
                filename = result.get('filename', 'Document')
                content = result.get('content', '')[:800]
                
                context_parts.append(f"\n=== DOCUMENT {i}: {filename} ===")
                context_parts.append(content)
                if len(result.get('content', '')) > 800:
                    context_parts.append("... [tronqué]")
            
            context_parts.append(f"\n🎯 UTILISE ces {len(search_results)} documents pour répondre précisément.")
            
            final_context = "\n".join(context_parts)
            
            logger.info(f"📝 Contexte généré pour {agent_name}: {len(search_results)} docs, {len(final_context)} chars")
            return final_context
            
        except Exception as e:
            logger.error(f"❌ Erreur contexte pour {agent_name}: {e}")
            return f"CONTEXTE RAG: Erreur - {str(e)}"
    
    def store_conversation(self, user_msg: str, agent_name: str, response: str, 
                          session_id: str = None, processing_time: float = 0.0,
                          rag_chunks: int = 0) -> bool:
        """
        Store conversation - Database V4
        """
        try:
            conv_id = self.db.store_conversation(
                user_msg=user_msg,
                agent_name=agent_name, 
                response=response,
                session_id=session_id,
                processing_time=processing_time,
                rag_chunks=rag_chunks
            )
            
            logger.info(f"💬 Conversation stockée: {agent_name}")
            return bool(conv_id)
            
        except Exception as e:
            logger.error(f"❌ Erreur store conversation: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        ✅ AJOUTÉ: get_status() pour backend FastAPI
        """
        status = {
            'initialized': self.initialized,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.initialized:
            try:
                health = self.db.health_check()
                status['database'] = health
                
                stats = self.db.get_stats()
                status['stats'] = stats
                
            except Exception as e:
                status['error'] = str(e)
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ✅ AJOUTÉ: get_stats() pour backend FastAPI
        """
        try:
            if self.initialized:
                db_stats = self.db.get_stats()
                return {
                    'database': db_stats,
                    'vector_store': {'available': False},
                    'rag_manager': {
                        'initialized': self.initialized,
                        'type': 'minimal_complete'
                    }
                }
            else:
                return {'error': 'RAG Manager non initialisé'}
        except Exception as e:
            logger.error(f"❌ Erreur get_stats: {e}")
            return {'error': str(e)}
    
    def store_document_v4(self, filename: str, content: str, file_path: str = None, 
                         file_hash: str = None, file_size: int = None) -> bool:
        """
        ✅ AJOUTÉ: store_document_v4() pour backend FastAPI
        Wrapper vers add_document()
        """
        try:
            success, message, result = self.add_document(
                filename=filename,
                file_path=file_path or f"documents/{filename}",
                content=content
            )
            
            if success:
                logger.info(f"✅ store_document_v4 réussi: {filename}")
                return True
            else:
                logger.error(f"❌ store_document_v4 échoué: {message}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur store_document_v4: {e}")
            return False
    
    def store_interaction_v4(self, session_id: str, user_message: str, 
                            agent_name: str, agent_response: str, 
                            processing_time: float = None, 
                            rag_chunks_used: int = 0) -> bool:
        """
        ✅ AJOUTÉ: store_interaction_v4() pour backend FastAPI  
        Wrapper vers store_conversation()
        """
        try:
            return self.store_conversation(
                user_msg=user_message,
                agent_name=agent_name,
                response=agent_response,
                session_id=session_id,
                processing_time=processing_time or 0.0,
                rag_chunks=rag_chunks_used
            )
        except Exception as e:
            logger.error(f"❌ Erreur store_interaction_v4: {e}")
            return False
    
    def hybrid_search(self, query: str, k_docs: int = None, k_interactions: int = None,
                     filename: str = None) -> Dict[str, Any]:
        """
        ✅ AJOUTÉ: hybrid_search() pour backend FastAPI
        Version simplifiée FTS5 seulement
        """
        try:
            k_docs = k_docs or 5
            search_results, stats = self.search(query, limit=k_docs)
            
            return {
                'results': search_results,
                'search_strategy': 'fts5_only',
                'total_sources': len(search_results),
                'confidence': 0.8 if search_results else 0.0,
                'metadata': {
                    'docs_count': len(search_results),
                    'interactions_count': 0,
                    'query_length': len(query),
                    'has_filename': bool(filename)
                }
            }
        except Exception as e:
            logger.error(f"❌ Erreur hybrid_search: {e}")
            return {
                'results': [],
                'search_strategy': 'error',
                'total_sources': 0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def get_conversations_context(self, limit: int = 10, session_id: str = None) -> List[Dict]:
        """
        ✅ AJOUTÉ: get_conversations_context() pour backend FastAPI
        """
        try:
            if self.initialized:
                conversations = self.db.get_conversations_history(session_id=session_id, limit=limit)
                logger.debug(f"💬 Contexte conversations: {len(conversations)} récupérées")
                return conversations
            else:
                return []
        except Exception as e:
            logger.error(f"❌ Erreur get_conversations_context: {e}")
            return []
    
    def cleanup(self):
        """
        ✅ AJOUTÉ: cleanup() pour backend FastAPI
        """
        try:
            logger.info("🧹 Cleanup RAG Manager (minimal - rien à faire)")
        except Exception as e:
            logger.error(f"❌ Erreur cleanup: {e}")


# === INSTANCE GLOBALE ===

_rag_instance = None

def get_rag_manager():
    """Factory simple"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGManagerV4Minimal()
    return _rag_instance

# Instance par défaut
rag_manager = get_rag_manager()


# === TEST ===
if __name__ == "__main__":
    print("=== TEST RAG MANAGER V4 COMPLET CORRIGÉ ===")
    
    rag = RAGManagerV4Minimal()
    print(f"Initialisé: {rag.initialized}")
    
    if rag.initialized:
        # Test get_stats (backend)
        stats = rag.get_stats()
        print(f"Stats: {stats.get('rag_manager', {}).get('type', 'unknown')}")
        
        # Test store_document_v4 (backend)
        success = rag.store_document_v4(
            filename="test_backend.txt",
            content="Test document pour backend FastAPI"
        )
        print(f"store_document_v4: {success}")
        
        if success:
            # Test recherche
            results, search_stats = rag.search("test")
            print(f"Recherche: {len(results)} résultats")
            
            # Test hybrid_search (backend)
            hybrid = rag.hybrid_search("test", k_docs=3)
            print(f"Hybrid search: {hybrid['total_sources']} sources")
            
            # Test contexte
            context = rag.get_context_for_agent("test", "TestAgent")
            print(f"Contexte: {len(context)} chars")
    
    print("\n✅ RAG MANAGER V4 COMPLET - TOUTES MÉTHODES BACKEND INCLUSES !")
    print("🎯 MÉTHODES AJOUTÉES POUR BACKEND:")
    print("   ✅ get_stats()")
    print("   ✅ store_document_v4()")
    print("   ✅ store_interaction_v4()")
    print("   ✅ hybrid_search()")
    print("   ✅ get_conversations_context()")
    print("   ✅ cleanup()")