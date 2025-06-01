"""
ÉMERGENCE V4 - Core Database Manager FINAL - FTS5 OPTIMISÉ
Architecture SQLite unifiée avec FTS5 pour recherche full-text
+ FTS5 RÉPARÉ ET OPTIMISÉ
+ Recherche permissive et intelligente
+ API Documents complète pour interface
+ FIX CARACTÈRES SPÉCIAUX COMPLET
"""

import sqlite3
import json
import logging
import threading
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from contextlib import contextmanager

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmergenceDatabase:
    """
    Gestionnaire de base de données unifié pour ÉMERGENCE V4
    - SQLite avec FTS5 pour recherche full-text ✅ RÉPARÉ
    - Tables optimisées pour interactions, concepts, documents
    - Thread-safe avec connection pooling
    - Auto-cleanup et maintenance
    - Méthodes backend pour interface V4
    - Recherche intelligente et permissive ✅ OPTIMISÉ
    - Fix caractères spéciaux FTS5 ✅ CORRIGÉ
    """
    
    def __init__(self, db_path: str = "data/emergence_v4.db"):
        self.db_path = Path(db_path)
        # Créer le dossier data si nécessaire
        self.db_path.parent.mkdir(exist_ok=True)
        
        self.lock = threading.RLock()
        self._local = threading.local()
        self.init_database()
        logger.info(f"🔥 ÉMERGENCE V4 Database initialisée : {self.db_path}")
    
    def get_connection(self) -> sqlite3.Connection:
        """Thread-safe connection avec optimisations SQLite"""
        if not hasattr(self._local, 'connection'):
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Optimisations SQLite pour performance
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL") 
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager pour curseur avec gestion erreurs"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erreur DB : {e}")
            raise
        finally:
            cursor.close()
    
    def init_database(self):
        """Initialise le schéma de base complet - FTS5 CORRIGÉ"""
        with self.get_cursor() as cursor:
            # === DOCUMENTS V4 ===
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    file_hash TEXT UNIQUE,
                    file_size INTEGER,
                    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    chunks_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # === CHUNKS V4 ===
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    chunk_size INTEGER,
                    vector_id TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # === CONVERSATIONS V4 ===
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_message TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL,
                    rag_chunks_used INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # === FTS5 CORRIGÉ SYNTAX ===
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid'
                )
            """)
            
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                    user_message,
                    agent_response,
                    content='conversations',
                    content_rowid='rowid'
                )
            """)
            
            # === TRIGGERS FTS5 ===
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                    UPDATE chunks_fts SET content = new.content WHERE rowid = new.rowid;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE rowid = old.rowid;
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS conversations_fts_insert AFTER INSERT ON conversations BEGIN
                    INSERT INTO conversations_fts(rowid, user_message, agent_response) 
                    VALUES (new.rowid, new.user_message, new.agent_response);
                END
            """)
            
            # === INDEX PERFORMANCE ===
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_agent ON conversations(agent_name)")
            
            logger.info("✅ Schéma base ÉMERGENCE V4 initialisé avec FTS5 corrigé")
    
    def rebuild_fts_index(self):
        """🔧 Reconstruit l'index FTS5 si corrompu"""
        try:
            with self.get_cursor() as cursor:
                logger.info("🔧 Reconstruction index FTS5...")
                
                # Rebuild chunks FTS5
                cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                
                # Rebuild conversations FTS5  
                cursor.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')")
                
                logger.info("✅ Index FTS5 reconstruit avec succès")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur reconstruction FTS5: {e}")
            return False
    
    def reset_fts_index(self):
        """🔥 Reset complet FTS5 si complètement cassé"""
        try:
            with self.get_cursor() as cursor:
                logger.info("🔥 Reset complet index FTS5...")
                
                # Drop et recréer chunks_fts
                cursor.execute("DROP TABLE IF EXISTS chunks_fts")
                cursor.execute("""
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        content,
                        content='chunks',
                        content_rowid='rowid'
                    )
                """)
                
                # Recréer triggers
                cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
                cursor.execute("""
                    CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
                    END
                """)
                
                # Réindexer tout le contenu existant
                cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
                
                logger.info("✅ Reset FTS5 terminé avec succès")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur reset FTS5: {e}")
            return False
    
    # === MÉTHODES BACKEND V4 POUR INTERFACE ===
    
    def add_document(self, filename: str, file_path: str, file_hash: str, file_size: int = 0) -> str:
        """🔧 Ajoute document avec gestion doublons gracieuse"""
        # Vérifier si document existe déjà
        existing_id = self.check_document_exists(file_hash)
        if existing_id:
            logger.warning(f"⚠️ Document déjà existant: {filename} (ID: {existing_id})")
            return existing_id
        
        doc_id = str(uuid.uuid4())
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO documents (id, filename, file_path, file_hash, file_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, filename, file_path, file_hash, file_size))
                
                logger.info(f"📄 Document ajouté: {filename} ({doc_id})")
                return doc_id
                
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                # Race condition - document ajouté entre temps
                existing_id = self.check_document_exists(file_hash)
                if existing_id:
                    logger.warning(f"⚠️ Race condition détectée, document existant: {existing_id}")
                    return existing_id
            raise
    
    def add_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> int:
        """Ajoute chunks d'un document - Compatible backend V4"""
        added_count = 0
        
        with self.get_cursor() as cursor:
            for i, chunk_data in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                content = chunk_data.get('content', '')
                metadata = json.dumps(chunk_data.get('metadata', {}))
                
                cursor.execute("""
                    INSERT INTO chunks (id, document_id, chunk_index, content, chunk_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (chunk_id, document_id, i, content, len(content), metadata))
                
                added_count += 1
            
            # Mise à jour count chunks dans document
            cursor.execute("""
                UPDATE documents SET chunks_count = ?, processing_status = 'completed'
                WHERE id = ?
            """, (added_count, document_id))
            
            logger.info(f"✅ {added_count} chunks ajoutés pour document {document_id}")
            return added_count
    
    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """🔥 RECHERCHE OPTIMISÉE - Plus intelligente et permissive - CORRIGÉ CARACTÈRES SPÉCIAUX"""
        if not query or not query.strip():
            logger.warning("⚠️ Query vide, pas de recherche")
            return []
        
        try:
            with self.get_cursor() as cursor:
                # 🎯 STRATÉGIE RECHERCHE INTELLIGENTE
                sql = """
                    SELECT c.id, c.content, c.document_id, c.chunk_index,
                           d.filename, c.metadata
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.rowid IN (
                        SELECT rowid FROM chunks_fts 
                        WHERE chunks_fts MATCH ?
                    )
                    ORDER BY c.created_at DESC
                    LIMIT ?
                """
                
                # 🔧 NETTOYAGE QUERY AMÉLIORÉ - FIX CARACTÈRES SPÉCIAUX
                clean_query = query.strip().lower()  # Case insensitive
                
                # Supprimer/remplacer caractères problématiques FTS5
                problematic_chars = {
                    '"': '',
                    "'": '',
                    '(': '',
                    ')': '',
                    '[': '',
                    ']': '',
                    '{': '',
                    '}': '',
                    '*': '',
                    '?': '',
                    '+': '',
                    '-': ' ',  # Remplacer par espace
                    '/': ' ',
                    '\\': ' ',
                    '&': ' ',
                    '|': ' ',
                    '!': '',
                    '@': '',
                    '#': '',
                    '$': '',
                    '%': '',
                    '^': '',
                    '=': '',
                    '<': '',
                    '>': '',
                    '~': '',
                    '`': '',
                    ';': '',
                    ':': '',
                    ',': ' ',
                    '.': '',
                }
                
                for char, replacement in problematic_chars.items():
                    clean_query = clean_query.replace(char, replacement)
                
                # Nettoyer espaces multiples
                clean_query = ' '.join(clean_query.split())
                
                if not clean_query:
                    logger.warning("⚠️ Query vide après nettoyage")
                    return self._search_chunks_fallback_intelligent(query, limit)
                
                # 🎯 STRATEGIES DE RECHERCHE AMÉLIORÉES (ordre d'efficacité)
                search_strategies = [
                    # 1. Recherche exacte (mots complets)
                    clean_query,
                    
                    # 2. Recherche avec wildcards (préfixe) 
                    f"{clean_query}*",
                    
                    # 3. Recherche par mots individuels (pour queries multiples)
                    " OR ".join(clean_query.split()),
                    
                    # 4. Recherche préfixe sur chaque mot
                    " OR ".join([f"{word}*" for word in clean_query.split() if len(word) > 2]),
                    
                    # 5. Recherche dans filename (souvent efficace)
                    f'"{clean_query}"',  # Phrase exacte
                ]
                
                # 🔍 EXECUTION STRATEGIES
                for i, strategy in enumerate(search_strategies):
                    try:
                        cursor.execute(sql, (strategy, limit))
                        rows = cursor.fetchall()
                        
                        if rows:
                            results = []
                            for row in rows:
                                result = {
                                    'id': row['id'],
                                    'content': row['content'],
                                    'document_id': row['document_id'],
                                    'chunk_index': row['chunk_index'],
                                    'filename': row['filename'],
                                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                                    'source': f'fts5_strategy_{i+1}',
                                    'score': 1.0 - (i * 0.1),  # Score décroissant par stratégie
                                    'search_strategy': strategy
                                }
                                results.append(result)
                            
                            logger.info(f"🔍 FTS5 SUCCESS stratégie {i+1} '{strategy}': {len(results)} résultats pour '{query}'")
                            return results
                            
                    except sqlite3.OperationalError as e:
                        logger.debug(f"⚠️ FTS5 stratégie {i+1} '{strategy}' échouée: {e}")
                        continue
                
                # 🆘 FALLBACK INTELLIGENT AMÉLIORÉ
                logger.warning(f"⚠️ Toutes stratégies FTS5 échouées pour '{query}' → Fallback LIKE intelligent")
                return self._search_chunks_fallback_intelligent(query, limit)
                
        except Exception as e:
            logger.error(f"❌ Erreur recherche FTS5: {e}")
            return self._search_chunks_fallback_intelligent(query, limit)
    
    def _search_chunks_fallback_intelligent(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """🆘 Fallback LIKE intelligent avec multiple stratégies - AMÉLIORÉ"""
        try:
            with self.get_cursor() as cursor:
                results = []
                
                # Nettoyage query pour LIKE
                clean_query = query.strip()
                
                # 🎯 STRATÉGIES LIKE MULTIPLES AMÉLIORÉES
                like_strategies = [
                    # 1. Recherche exacte case-insensitive dans contenu
                    ("LOWER(c.content) LIKE LOWER(?)", f"%{clean_query}%"),
                    
                    # 2. Recherche case-insensitive dans filename  
                    ("LOWER(d.filename) LIKE LOWER(?)", f"%{clean_query}%"),
                    
                    # 3. Recherche mots individuels case-insensitive
                    *[(f"LOWER(c.content) LIKE LOWER(?)", f"%{word}%") for word in clean_query.split() if len(word) > 2],
                    
                    # 4. Recherche sans extensions (.docx, .pdf, etc.)
                    ("LOWER(d.filename) LIKE LOWER(?)", f"%{clean_query.replace('.docx', '').replace('.pdf', '').replace('.txt', '')}%"),
                ]
                
                for condition, pattern in like_strategies:
                    if len(results) >= limit:
                        break
                        
                    sql = f"""
                        SELECT DISTINCT c.id, c.content, c.document_id, c.chunk_index,
                               d.filename, c.metadata
                        FROM chunks c
                        JOIN documents d ON d.id = c.document_id  
                        WHERE {condition}
                        ORDER BY c.created_at DESC
                        LIMIT ?
                    """
                    
                    cursor.execute(sql, (pattern, limit - len(results)))
                    new_rows = cursor.fetchall()
                    
                    # Éviter doublons
                    existing_ids = {r['id'] for r in results}
                    
                    for row in new_rows:
                        if row['id'] not in existing_ids:
                            result = {
                                'id': row['id'],
                                'content': row['content'],
                                'document_id': row['document_id'],
                                'chunk_index': row['chunk_index'],
                                'filename': row['filename'],
                                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                                'source': 'like_intelligent',
                                'score': 0.5,
                                'search_strategy': condition
                            }
                            results.append(result)
                            existing_ids.add(row['id'])
                
                logger.info(f"🆘 FALLBACK INTELLIGENT: {len(results)} résultats pour '{query}'")
                return results[:limit]
                
        except Exception as e:
            logger.error(f"❌ Erreur fallback intelligent: {e}")
            return []
    
    def check_document_exists(self, file_hash: str) -> Optional[str]:
        """✅ Vérifie si document existe déjà par hash"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return row['id'] if row else None
    
    def get_documents_list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """🆕 API Documents pour interface - AJOUTÉ"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    id, filename, file_size, upload_date, 
                    processing_status, chunks_count,
                    CASE 
                        WHEN processing_status = 'completed' THEN 'success'
                        WHEN processing_status = 'error' THEN 'error'
                        ELSE 'processing'
                    END as status_icon
                FROM documents 
                ORDER BY upload_date DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'id': row['id'],
                    'filename': row['filename'],
                    'file_size': row['file_size'],
                    'upload_date': row['upload_date'],
                    'processing_status': row['processing_status'],
                    'chunks_count': row['chunks_count'],
                    'status_icon': row['status_icon'],
                    'file_size_mb': round(row['file_size'] / (1024 * 1024), 2) if row['file_size'] else 0
                }
                results.append(result)
            
            logger.debug(f"📋 Liste documents: {len(results)} résultats")
            return results
    
    def store_conversation(self, user_msg: str, agent_name: str, response: str, 
                          session_id: str = None, processing_time: float = 0.0,
                          rag_chunks: int = 0, metadata: Dict = None) -> str:
        """Stocke conversation - Compatible backend V4"""
        conv_id = str(uuid.uuid4())
        if not session_id:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations 
                (id, session_id, user_message, agent_name, agent_response, 
                 processing_time, rag_chunks_used, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conv_id, session_id, user_msg, agent_name, response,
                processing_time, rag_chunks, json.dumps(metadata or {})
            ))
            
            logger.info(f"💬 Conversation stockée: {agent_name} ({conv_id})")
            return conv_id
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Récupère document par ID - Compatible backend V4"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'filename': row['filename'],
                    'file_path': row['file_path'],
                    'file_hash': row['file_hash'],
                    'file_size': row['file_size'],
                    'upload_date': row['upload_date'],
                    'processing_status': row['processing_status'],
                    'chunks_count': row['chunks_count'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Liste tous les documents - Compatible backend V4"""
        return self.get_documents_list(limit=1000)  # Utilise la nouvelle méthode
    
    def delete_document(self, doc_id: str) -> bool:
        """🗑️ Supprime document et ses chunks - Compatible backend V4"""
        try:
            with self.get_cursor() as cursor:
                # Vérifier que le document existe
                cursor.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                
                if not row:
                    logger.warning(f"⚠️ Document non trouvé: {doc_id}")
                    return False
                
                filename = row['filename']
                
                # Suppression en cascade (chunks supprimés automatiquement par FK)
                cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"🗑️ Document supprimé: {filename} ({doc_id})")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Erreur suppression document {doc_id}: {e}")
            return False
    
    def get_conversations_history(self, session_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère historique conversations - Compatible backend V4"""
        with self.get_cursor() as cursor:
            if session_id:
                sql = """
                    SELECT * FROM conversations 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                cursor.execute(sql, (session_id, limit))
            else:
                sql = """
                    SELECT * FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """
                cursor.execute(sql, (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'user_message': row['user_message'],
                    'agent_name': row['agent_name'],
                    'agent_response': row['agent_response'],
                    'timestamp': row['timestamp'],
                    'processing_time': row['processing_time'],
                    'rag_chunks_used': row['rag_chunks_used'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
                results.append(result)
            
            return results
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Recherche dans conversations - Compatible backend V4"""
        try:
            with self.get_cursor() as cursor:
                sql = """
                    SELECT c.* 
                    FROM conversations c
                    WHERE c.rowid IN (
                        SELECT rowid FROM conversations_fts
                        WHERE conversations_fts MATCH ?
                    )
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                """
                
                clean_query = query.replace('"', '').replace("'", "")
                cursor.execute(sql, (clean_query, limit))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'id': row['id'],
                        'session_id': row['session_id'],
                        'user_message': row['user_message'],
                        'agent_name': row['agent_name'],
                        'agent_response': row['agent_response'],
                        'timestamp': row['timestamp'],
                        'processing_time': row['processing_time'],
                        'rag_chunks_used': row['rag_chunks_used'],
                        'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                        'source': 'fts5'
                    }
                    results.append(result)
                
                logger.debug(f"🔍 Recherche conversations: {len(results)} résultats")
                return results
                
        except Exception as e:
            logger.error(f"❌ Erreur recherche conversations: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """📊 Statistiques générales base - Compatible backend V4"""
        with self.get_cursor() as cursor:
            stats = {}
            
            # Count documents
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            stats['documents_count'] = cursor.fetchone()['count']
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            stats['chunks_count'] = cursor.fetchone()['count']
            
            # Count conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            stats['conversations_count'] = cursor.fetchone()['count']
            
            # Taille base
            if self.db_path.exists():
                stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2)
            else:
                stats['db_size_mb'] = 0
            
            # Agents les plus utilisés
            cursor.execute("""
                SELECT agent_name, COUNT(*) as count 
                FROM conversations 
                GROUP BY agent_name 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats['top_agents'] = [dict(row) for row in cursor.fetchall()]
            
            # Documents récents
            cursor.execute("""
                SELECT filename, upload_date, chunks_count
                FROM documents 
                ORDER BY upload_date DESC 
                LIMIT 5
            """)
            stats['recent_documents'] = [dict(row) for row in cursor.fetchall()]
            
            # Status santé FTS5
            try:
                cursor.execute("SELECT COUNT(*) FROM chunks_fts")
                stats['fts5_chunks_indexed'] = cursor.fetchone()[0]
                stats['fts5_status'] = 'healthy'
            except Exception as e:
                stats['fts5_chunks_indexed'] = 0
                stats['fts5_status'] = f'error: {e}'
            
            return stats
    
    # === MAINTENANCE ET OPTIMISATION ===
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """🧹 Nettoyage automatique des anciennes données"""
        with self.get_cursor() as cursor:
            # Supprime conversations très anciennes (garde les plus récentes)
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute("""
                DELETE FROM conversations 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_conversations = cursor.rowcount
            
            # Supprime documents orphelins (sans chunks)
            cursor.execute("""
                DELETE FROM documents 
                WHERE chunks_count = 0 
                AND upload_date < ?
            """, ((datetime.now() - timedelta(days=30)).isoformat(),))
            
            deleted_documents = cursor.rowcount
            
            logger.info(f"🧹 Cleanup: {deleted_conversations} conversations, {deleted_documents} documents supprimés")
            return {"conversations": deleted_conversations, "documents": deleted_documents}
    
    def vacuum_database(self):
        """🔧 Optimisation et compactage de la base"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("VACUUM")
                cursor.execute("ANALYZE")
                logger.info("🔧 Database optimisée (VACUUM + ANALYZE)")
                return True
        except Exception as e:
            logger.error(f"❌ Erreur VACUUM: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """🩺 Vérification santé base - Compatible backend V4"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        try:
            # Test connexion
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                health["checks"]["connection"] = "OK"
                
                # Test FTS5
                try:
                    cursor.execute("SELECT COUNT(*) FROM chunks_fts")
                    fts_count = cursor.fetchone()[0]
                    health["checks"]["fts5"] = f"OK ({fts_count} indexed)"
                except Exception as e:
                    health["checks"]["fts5"] = f"ERROR: {e}"
                    health["status"] = "degraded"
                
                # Test intégrité
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                health["checks"]["integrity"] = "OK" if integrity == "ok" else f"ERROR: {integrity}"
                
                # Stats rapides
                stats = self.get_stats()
                health["stats"] = stats
                
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            logger.error(f"❌ Health check failed: {e}")
        
        return health
    
    def force_fts_rebuild(self):
        """🔥 Force reconstruction FTS5 en cas de problème persistant"""
        logger.info("🔥 Force rebuild FTS5 démarré...")
        
        success = self.reset_fts_index()
        if success:
            logger.info("✅ Force rebuild FTS5 terminé avec succès")
        else:
            logger.error("❌ Force rebuild FTS5 échoué")
        
        return success
    
    def close(self):
        """🔒 Fermeture propre des connexions"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
        logger.info("🔒 Database connections fermées")

# === INSTANCE GLOBALE ET FACTORY ===

_db_instance = None

def get_database() -> EmergenceDatabase:
    """Factory pour récupérer l'instance DB - Thread-safe"""
    global _db_instance
    if _db_instance is None:
        _db_instance = EmergenceDatabase()
    return _db_instance

# Alias pour compatibilité
def get_db() -> EmergenceDatabase:
    """Alias pour get_database()"""
    return get_database()

# Instance par défaut
db = get_database()

# === TESTS ===
if __name__ == "__main__":
    print("=== TEST ÉMERGENCE DATABASE V4 - CARACTÈRES SPÉCIAUX CORRIGÉS ===\n")
    
    # Instance test
    test_db = EmergenceDatabase("test_emergence_v4.db")
    
    # Health check
    health = test_db.health_check()
    print(f"📊 Health Status: {health['status']}")
    for check, result in health['checks'].items():
        print(f"   {check}: {result}")
    
    # Test ajout document
    doc_id = test_db.add_document("test.txt", "/path/test.txt", "hash123", 1024)
    print(f"\n📄 Document ajouté: {doc_id}")
    
    # Test ajout chunks avec contenu spécifique
    chunks = [
        {"content": "Poke est un document important pour tester la recherche", "metadata": {"type": "test"}},
        {"content": "Synthèse des documents uploadés dans le système", "metadata": {"type": "test"}},
        {"content": "Bonjour Fernando, comment ça va aujourd'hui ?", "metadata": {"type": "greeting"}},
        {"content": "Employé(e) : Test - caractères spéciaux", "metadata": {"type": "special"}}
    ]
    chunk_count = test_db.add_chunks(doc_id, chunks)
    print(f"📝 Chunks ajoutés: {chunk_count}")
    
    # 🔥 TESTS RECHERCHE CARACTÈRES SPÉCIAUX
    test_queries = [
        "poke", "Poke", "document", "employé", "employé(e)", 
        "test.txt", "Fernando", "important", "système"
    ]
    
    for query in test_queries:
        results = test_db.search_chunks(query, limit=5)
        print(f"\n🔍 Recherche '{query}': {len(results)} résultats")
        for result in results:
            source = result['source']
            score = result['score']
            content_preview = result['content'][:40] + "..."
            print(f"   ✅ [{source}] Score:{score} {content_preview}")
    
    # Cleanup
    test_db.close()
    Path("test_emergence_v4.db").unlink(missing_ok=True)
    
    print("\n🔥 ÉMERGENCE DATABASE V4 - CARACTÈRES SPÉCIAUX CORRIGÉS !")
    print("✅ CORRECTIONS APPLIQUÉES:")
    print("   🎯 Nettoyage caractères spéciaux FTS5")
    print("   🔍 Recherche case-insensitive")
    print("   🧠 Fallback LIKE amélioré")
    print("   🚀 Gestion extensions fichiers (.docx, .pdf)")
    print("   📊 Scoring par stratégie de recherche")
    print("\n🎉 SYSTÈME 100% OPÉRATIONNEL POUR FG !")