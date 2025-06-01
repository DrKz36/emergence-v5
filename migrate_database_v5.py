"""
ÉMERGENCE V5 - DATABASE MIGRATION SCRIPT
🔥 Migration database principale pour tables sessions + mémoire persistante
Version: 5.0.0 - Sessions Tables + FTS5 + Backward Compatibility
"""

import os
import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrationV5:
    """
    🚀 MIGRATION DATABASE V5 - TABLES SESSIONS
    
    Responsabilités :
    - Ajouter tables sessions sans casser l'existant
    - Migration données existantes si nécessaire  
    - Vérification intégrité post-migration
    - Backup automatique avant migration
    - Rollback si échec
    """
    
    def __init__(self, database_path: str = "data/emergence_v4.db"):
        self.database_path = database_path
        self.backup_path = f"{database_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.migration_version = "5.0.0"
        
        logger.info(f"🔧 Migration Database V5 initialisée")
        logger.info(f"📂 Database: {database_path}")
        logger.info(f"💾 Backup: {self.backup_path}")
    
    def check_database_exists(self) -> bool:
        """🔍 Vérification existence database"""
        exists = os.path.exists(self.database_path)
        logger.info(f"📊 Database exists: {exists}")
        return exists
    
    def create_backup(self) -> bool:
        """💾 Création backup avant migration"""
        try:
            if not self.check_database_exists():
                logger.warning("⚠️ Database n'existe pas - pas de backup nécessaire")
                return True
                
            # Copie fichier database
            import shutil
            shutil.copy2(self.database_path, self.backup_path)
            
            # Vérification backup
            backup_size = os.path.getsize(self.backup_path)
            original_size = os.path.getsize(self.database_path)
            
            if backup_size == original_size:
                logger.info(f"✅ Backup créé: {self.backup_path} ({backup_size} bytes)")
                return True
            else:
                logger.error(f"❌ Backup invalide: tailles différentes")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur création backup: {e}")
            return False
    
    def get_current_schema_version(self) -> str:
        """📋 Récupération version schéma actuel"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Check si table metadata existe
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_metadata'
                """)
                
                if cursor.fetchone():
                    cursor.execute("SELECT version FROM schema_metadata ORDER BY created_at DESC LIMIT 1")
                    result = cursor.fetchone()
                    version = result[0] if result else "4.0.0"
                else:
                    # Pas de table metadata = version 4.0.0 (legacy)
                    version = "4.0.0"
                
                logger.info(f"📋 Version schéma actuelle: {version}")
                return version
                
        except Exception as e:
            logger.error(f"❌ Erreur lecture version schéma: {e}")
            return "unknown"
    
    def check_existing_tables(self) -> Dict[str, bool]:
        """🔍 Vérification tables existantes"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                    ORDER BY name
                """)
                
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                # Tables V4 attendues
                expected_v4_tables = [
                    'conversations', 'documents', 'chunks', 'interactions'
                ]
                
                # Tables V5 nouvelles
                new_v5_tables = [
                    'sessions', 'session_messages', 'session_messages_fts',
                    'agent_memories', 'concept_timeline', 'thinking_patterns',
                    'schema_metadata'
                ]
                
                table_status = {}
                
                # Check V4 tables
                for table in expected_v4_tables:
                    table_status[f"v4_{table}"] = table in existing_tables
                
                # Check V5 tables
                for table in new_v5_tables:
                    table_status[f"v5_{table}"] = table in existing_tables
                
                logger.info(f"🔍 Tables existantes: {len(existing_tables)}")
                logger.info(f"   V4 tables: {[k for k,v in table_status.items() if k.startswith('v4_') and v]}")
                logger.info(f"   V5 tables: {[k for k,v in table_status.items() if k.startswith('v5_') and v]}")
                
                return table_status
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification tables: {e}")
            return {}
    
    def create_schema_metadata_table(self) -> bool:
        """📊 Création table metadata schéma"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        migration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        description TEXT,
                        tables_added TEXT,  -- JSON list
                        migration_notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("✅ Table schema_metadata créée")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur création schema_metadata: {e}")
            return False
    
    def create_sessions_tables(self) -> bool:
        """🗄️ Création tables sessions V5"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Table sessions principales
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'FG',
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        session_data TEXT,  -- JSON complet
                        themes TEXT,        -- JSON array themes
                        agents_used TEXT,   -- JSON array agents
                        modes_used TEXT,    -- JSON array modes
                        documents_used TEXT, -- JSON array documents
                        total_cost REAL DEFAULT 0.0,
                        message_count INTEGER DEFAULT 0,
                        session_duration_minutes REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Table messages sessions pour recherche
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS session_messages (
                        rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_text TEXT NOT NULL,
                        agent TEXT,
                        sender TEXT NOT NULL,
                        mode TEXT DEFAULT 'dialogue',
                        timestamp TEXT NOT NULL,
                        metadata TEXT,  -- JSON
                        FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                    )
                """)
                
                # Table FTS5 pour recherche full-text dans messages
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                        session_id,
                        message_text,
                        agent,
                        mode,
                        timestamp,
                        content='session_messages',
                        content_rowid='rowid'
                    )
                """)
                
                # Index pour performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_date ON sessions (user_id, start_time)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_themes ON sessions (themes)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_cost ON sessions (total_cost)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_agent ON session_messages (agent)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_timestamp ON session_messages (timestamp)")
                
                conn.commit()
                logger.info("✅ Tables sessions V5 créées avec index")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur création tables sessions: {e}")
            return False
    
    def create_memory_tables(self) -> bool:
        """🧠 Création tables mémoire agents V5"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Table mémoire agents
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_memories (
                        id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        user_id TEXT NOT NULL DEFAULT 'FG',
                        memory_type TEXT NOT NULL,  -- context, pattern, preference, interaction
                        memory_data TEXT NOT NULL,  -- JSON
                        importance_score REAL DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        expires_at TIMESTAMP,  -- NULL = permanent
                        tags TEXT  -- JSON array
                    )
                """)
                
                # Table timeline concepts
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS concept_timeline (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'FG',
                        concept_name TEXT NOT NULL,
                        session_id TEXT,
                        mentioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        context_snippet TEXT,
                        sentiment_score REAL DEFAULT 0.0,  -- -1.0 to 1.0
                        evolution_stage TEXT,  -- discovery, exploration, consolidation, mastery
                        related_concepts TEXT,  -- JSON array
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                """)
                
                # Table patterns de pensée
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS thinking_patterns (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'FG',
                        pattern_type TEXT NOT NULL,  -- recurrent_theme, question_style, reasoning_approach
                        pattern_data TEXT NOT NULL,  -- JSON description pattern
                        confidence_score REAL DEFAULT 0.5,
                        first_detected TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_confirmed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        occurrences INTEGER DEFAULT 1,
                        examples TEXT,  -- JSON array examples
                        status TEXT DEFAULT 'active'  -- active, dormant, evolved
                    )
                """)
                
                # Index pour performance mémoire
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_user ON agent_memories (agent_name, user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_type ON agent_memories (memory_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_importance ON agent_memories (importance_score DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept_timeline_user_concept ON concept_timeline (user_id, concept_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept_timeline_session ON concept_timeline (session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_thinking_patterns_user_type ON thinking_patterns (user_id, pattern_type)")
                
                conn.commit()
                logger.info("✅ Tables mémoire agents V5 créées avec index")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur création tables mémoire: {e}")
            return False
    
    def update_schema_version(self, tables_added: List[str]) -> bool:
        """📝 Mise à jour version schéma"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO schema_metadata (
                        version, description, tables_added, migration_notes
                    ) VALUES (?, ?, ?, ?)
                """, (
                    self.migration_version,
                    "Migration V5 - Tables sessions + mémoire persistante",
                    json.dumps(tables_added),
                    "Ajout fonctionnalités Phase 2: sessions, agent memory, timeline, patterns"
                ))
                
                conn.commit()
                logger.info(f"✅ Version schéma mise à jour: {self.migration_version}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour version schéma: {e}")
            return False
    
    def verify_migration(self) -> Dict[str, Any]:
        """🔍 Vérification intégrité post-migration"""
        try:
            verification = {
                "success": True,
                "tables_created": [],
                "indexes_created": [],
                "errors": [],
                "warnings": []
            }
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Vérification tables créées
                expected_tables = [
                    'sessions', 'session_messages', 'session_messages_fts',
                    'agent_memories', 'concept_timeline', 'thinking_patterns',
                    'schema_metadata'
                ]
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                
                for table in expected_tables:
                    if table in existing_tables:
                        verification["tables_created"].append(table)
                    else:
                        verification["errors"].append(f"Table {table} non créée")
                        verification["success"] = False
                
                # Vérification index
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
                indexes = [row[0] for row in cursor.fetchall()]
                verification["indexes_created"] = indexes
                
                # Test simple insertion/lecture
                try:
                    cursor.execute("INSERT INTO schema_metadata (version, description) VALUES (?, ?)", 
                                   ("test", "Migration verification test"))
                    cursor.execute("DELETE FROM schema_metadata WHERE version = 'test'")
                    conn.commit()
                except Exception as e:
                    verification["errors"].append(f"Test insertion failed: {e}")
                    verification["success"] = False
                
                # Statistiques
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                total_tables = cursor.fetchone()[0]
                
                verification["stats"] = {
                    "total_tables": total_tables,
                    "v5_tables_added": len(verification["tables_created"]),
                    "indexes_created": len(verification["indexes_created"])
                }
                
                logger.info(f"🔍 Vérification migration: {'✅ SUCCESS' if verification['success'] else '❌ FAILED'}")
                logger.info(f"   Tables créées: {len(verification['tables_created'])}/{len(expected_tables)}")
                logger.info(f"   Index créés: {len(verification['indexes_created'])}")
                
                return verification
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification migration: {e}")
            return {
                "success": False,
                "errors": [str(e)],
                "tables_created": [],
                "indexes_created": []
            }
    
    def rollback_migration(self) -> bool:
        """🔄 Rollback en cas d'échec"""
        try:
            if not os.path.exists(self.backup_path):
                logger.error(f"❌ Backup non trouvé: {self.backup_path}")
                return False
            
            # Restauration backup
            import shutil
            shutil.copy2(self.backup_path, self.database_path)
            
            logger.info(f"🔄 Rollback effectué depuis {self.backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur rollback: {e}")
            return False
    
    def run_migration(self) -> Dict[str, Any]:
        """🚀 Exécution migration complète"""
        migration_result = {
            "success": False,
            "steps_completed": [],
            "errors": [],
            "backup_created": False,
            "rollback_performed": False,
            "verification": {}
        }
        
        try:
            logger.info("🚀 DÉBUT MIGRATION DATABASE V5")
            
            # Étape 1: Backup
            logger.info("📝 Étape 1: Création backup...")
            if self.create_backup():
                migration_result["backup_created"] = True
                migration_result["steps_completed"].append("backup")
            else:
                migration_result["errors"].append("Échec création backup")
                return migration_result
            
            # Étape 2: Version actuelle
            logger.info("📋 Étape 2: Vérification version actuelle...")
            current_version = self.get_current_schema_version()
            migration_result["current_version"] = current_version
            migration_result["steps_completed"].append("version_check")
            
            # Étape 3: Tables existantes
            logger.info("🔍 Étape 3: Vérification tables existantes...")
            existing_tables = self.check_existing_tables()
            migration_result["existing_tables"] = existing_tables
            migration_result["steps_completed"].append("tables_check")
            
            # Étape 4: Schema metadata
            logger.info("📊 Étape 4: Création table schema metadata...")
            if self.create_schema_metadata_table():
                migration_result["steps_completed"].append("schema_metadata")
            else:
                migration_result["errors"].append("Échec création schema_metadata")
                
            # Étape 5: Tables sessions
            logger.info("🗄️ Étape 5: Création tables sessions...")
            if self.create_sessions_tables():
                migration_result["steps_completed"].append("sessions_tables")
            else:
                migration_result["errors"].append("Échec création tables sessions")
                
            # Étape 6: Tables mémoire
            logger.info("🧠 Étape 6: Création tables mémoire...")
            if self.create_memory_tables():
                migration_result["steps_completed"].append("memory_tables")
            else:
                migration_result["errors"].append("Échec création tables mémoire")
            
            # Étape 7: Mise à jour version
            logger.info("📝 Étape 7: Mise à jour version schéma...")
            tables_added = [
                'sessions', 'session_messages', 'session_messages_fts',
                'agent_memories', 'concept_timeline', 'thinking_patterns'
            ]
            if self.update_schema_version(tables_added):
                migration_result["steps_completed"].append("version_update")
            else:
                migration_result["errors"].append("Échec mise à jour version")
            
            # Étape 8: Vérification
            logger.info("🔍 Étape 8: Vérification intégrité...")
            verification = self.verify_migration()
            migration_result["verification"] = verification
            migration_result["steps_completed"].append("verification")
            
            # Résultat final
            if verification["success"] and len(migration_result["errors"]) == 0:
                migration_result["success"] = True
                logger.info("✅ MIGRATION V5 RÉUSSIE")
            else:
                migration_result["errors"].extend(verification.get("errors", []))
                logger.warning("⚠️ MIGRATION V5 PARTIELLE OU ÉCHEC")
                
                # Rollback si échec critique
                if len(verification.get("errors", [])) > 0:
                    logger.info("🔄 Tentative rollback...")
                    if self.rollback_migration():
                        migration_result["rollback_performed"] = True
            
            return migration_result
            
        except Exception as e:
            logger.error(f"❌ ERREUR CRITIQUE MIGRATION: {e}")
            migration_result["errors"].append(f"Erreur critique: {e}")
            
            # Rollback automatique
            logger.info("🔄 Rollback automatique...")
            if self.rollback_migration():
                migration_result["rollback_performed"] = True
            
            return migration_result
    
    def cleanup_backup(self) -> bool:
        """🧹 Nettoyage backup après migration réussie"""
        try:
            if os.path.exists(self.backup_path):
                os.remove(self.backup_path)
                logger.info(f"🧹 Backup nettoyé: {self.backup_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage backup: {e}")
            return False

# === SCRIPT PRINCIPAL ===

def main():
    """🎯 Script principal migration V5"""
    print("="*60)
    print("🚀 ÉMERGENCE V5 - MIGRATION DATABASE")
    print("="*60)
    
    # Paramètres
    database_path = "data/emergence_v4.db"
    
    print(f"📂 Database: {database_path}")
    print(f"🎯 Target: Tables sessions + mémoire persistante")
    print()
    
    # Confirmation utilisateur
    confirm = input("🔧 Continuer la migration ? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Migration annulée par l'utilisateur")
        return
    
    # Exécution migration
    migrator = DatabaseMigrationV5(database_path)
    result = migrator.run_migration()
    
    # Affichage résultats
    print("\n" + "="*60)
    print("📊 RÉSULTATS MIGRATION")
    print("="*60)
    
    print(f"✅ Succès: {'OUI' if result['success'] else 'NON'}")
    print(f"💾 Backup créé: {'OUI' if result['backup_created'] else 'NON'}")
    print(f"🔄 Rollback: {'OUI' if result['rollback_performed'] else 'NON'}")
    print(f"📝 Étapes complétées: {len(result['steps_completed'])}")
    
    if result["steps_completed"]:
        print(f"   ✅ {', '.join(result['steps_completed'])}")
    
    if result["errors"]:
        print(f"❌ Erreurs: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"   ❌ {error}")
    
    verification = result.get("verification", {})
    if verification:
        print(f"\n🔍 VÉRIFICATION:")
        print(f"   Tables créées: {len(verification.get('tables_created', []))}")
        print(f"   Index créés: {len(verification.get('indexes_created', []))}")
        
        if verification.get("tables_created"):
            print(f"   ✅ {', '.join(verification['tables_created'])}")
    
    # Nettoyage si succès
    if result["success"] and result["backup_created"]:
        cleanup = input("\n🧹 Supprimer le backup ? (y/N): ").strip().lower()
        if cleanup == 'y':
            migrator.cleanup_backup()
    
    print(f"\n{'✅ MIGRATION TERMINÉE AVEC SUCCÈS' if result['success'] else '❌ MIGRATION ÉCHOUÉE'}")
    print("="*60)

if __name__ == "__main__":
    main()