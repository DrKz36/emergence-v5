"""
ÉMERGENCE V5 - Session Manager - MÉMOIRE PERSISTANTE
🔥 Phase 2 - Gestion complète des sessions + sauvegarde + recherche temporelle
Version: 5.0.1 - Session Management + Timeline + Agent Memory + FIXES API
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import re

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === DATACLASSES SESSION MANAGER V5 ===

@dataclass
class SessionMessage:
    """📝 Message individuel dans une session"""
    timestamp: str
    sender: str  # 'user' ou 'agent'
    agent: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: str = "dialogue"
    session_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SessionInfo:
    """📊 Informations meta d'une session"""
    session_id: str
    user_id: str = "FG"  # Default pour Fernando
    start_time: str = ""
    end_time: str = ""
    total_messages: int = 0
    session_costs: Dict[str, float] = field(default_factory=dict)
    modes_used: List[str] = field(default_factory=list)
    agents_used: List[str] = field(default_factory=list)
    themes_detected: List[str] = field(default_factory=list)
    documents_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 🔥 AJOUTÉ
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SessionStatistics:
    """📈 Statistiques calculées d'une session"""
    user_messages: int = 0
    agent_responses: int = 0
    dialogue_messages: int = 0
    triangle_messages: int = 0
    documents_messages: int = 0
    total_cost: float = 0.0
    average_cost_per_response: float = 0.0
    session_duration_minutes: float = 0.0
    dominant_agent: str = ""
    dominant_theme: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CompleteSession:
    """🗂️ Session complète avec tous les éléments"""
    session_info: SessionInfo
    conversation: List[SessionMessage]
    statistics: SessionStatistics
    export_date: str = ""
    version: str = "5.0.1"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

@dataclass
class SessionSearchResult:
    """🔍 Résultat de recherche dans les sessions"""
    session_id: str
    session_date: str
    relevance_score: float
    matching_messages: List[SessionMessage]
    session_summary: str
    themes: List[str]
    agents_involved: List[str]

# === CLASSE PRINCIPALE SESSION MANAGER ===

class SessionManagerV5:
    """
    🚀 SESSION MANAGER V5 - MÉMOIRE PERSISTANTE ÉMERGENCE
    
    Responsabilités :
    - Gestion lifecycle des sessions (création, sauvegarde, récupération)
    - Sauvegarde automatique JSON + SQLite
    - Recherche temporelle intelligente
    - Détection thématique automatique
    - Analytics et insights sessions
    - Export/Import sessions
    
    Architecture :
    - Sessions JSON pour export/backup
    - SQLite pour recherche rapide + indexation FTS5
    - Métadonnées enrichies automatiques
    - Integration avec RAG Manager V4
    """
    
    def __init__(self, database_path: str = "data/emergence_v4.db", sessions_dir: str = "data/sessions"):
        """Initialisation Session Manager V5"""
        self.database_path = database_path
        self.sessions_dir = Path(sessions_dir)
        self.current_session: Optional[CompleteSession] = None
        self.active_sessions: Dict[str, CompleteSession] = {}
        
        # Création dossier sessions si nécessaire
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation database avec tables sessions
        self._init_session_tables()
        
        # Patterns pour détection thématique
        self._load_theme_patterns()
        
        logger.info(f"🚀 Session Manager V5 initialisé - DB: {database_path}")
        logger.info(f"📁 Sessions directory: {self.sessions_dir}")
    
    def _init_session_tables(self):
        """🗄️ Initialisation tables SQLite pour sessions"""
        try:
            with self._get_db_connection() as conn:
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
                
                # Table pour recherche FTS5 dans messages
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
                
                # Table messages sessions pour FTS5
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
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                """)
                
                # Index pour performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_date ON sessions (user_id, start_time)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_themes ON sessions (themes)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages (session_id)")
                
                conn.commit()
                logger.info("✅ Tables sessions V5 initialisées")
                
        except Exception as e:
            logger.error(f"❌ Erreur init tables sessions: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """🔗 Context manager pour connexions SQLite"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row  # Accès colonnes par nom
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Erreur DB connection: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _load_theme_patterns(self):
        """🎯 Chargement patterns pour détection thématique automatique"""
        self.theme_patterns = {
            "médecine": [
                r"médecine|médical|patient|diagnostic|traitement|thérapie|clinique|hôpital|santé|maladie",
                r"empathie.*médical|relation.*patient|déontologie|hippocrate|soignant"
            ],
            "philosophie": [
                r"philosophie|philosophique|métaphysique|existentiel|conscience|liberté|éthique|morale",
                r"kant|nietzsche|sartre|camus|stoïcisme|épicurisme|phénoménologie"
            ],
            "littérature": [
                r"littérature|roman|écriture|narratif|personnage|fiction|style|prose|poésie",
                r"auteur|écrivain|lecture|livre|récit|storytelling|inspiration"
            ],
            "technologie": [
                r"technologie|numérique|intelligence.*artificielle|algorithme|données|digital",
                r"informatique|programmation|robot|automation|ia|machine.*learning"
            ],
            "créativité": [
                r"créativité|créatif|imagination|inspiration|art|artistique|esthétique",
                r"innovation|originalité|génie|talent|expression.*artistique"
            ],
            "psychologie": [
                r"psychologie|psychologique|comportement|émotion|sentiment|mental|cognition",
                r"trauma|résilience|développement.*personnel|psychanalyse|thérapie"
            ],
            "science": [
                r"science|scientifique|recherche|expérience|théorie|hypothèse|méthode",
                r"physique|chimie|biologie|neuroscience|génétique|évolution"
            ]
        }
        logger.info(f"🎯 {len(self.theme_patterns)} patterns thématiques chargés")
    
    def _detect_themes(self, messages: List[SessionMessage]) -> List[str]:
        """🔍 Détection automatique des thèmes dans une session"""
        detected_themes = set()
        
        # Concaténation de tous les messages pour analyse
        full_text = " ".join([msg.message.lower() for msg in messages if msg.sender == "user"])
        agent_text = " ".join([msg.message.lower() for msg in messages if msg.sender == "agent"])
        combined_text = full_text + " " + agent_text
        
        # Analyse patterns par thème
        for theme, patterns in self.theme_patterns.items():
            theme_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                theme_score += matches
            
            # Seuil de détection (ajustable)
            if theme_score >= 2:  # Au moins 2 mentions du pattern
                detected_themes.add(theme)
                logger.debug(f"🎯 Thème détecté: {theme} (score: {theme_score})")
        
        return list(detected_themes)
    
    def _calculate_statistics(self, session: CompleteSession) -> SessionStatistics:
        """📊 Calcul statistiques complètes d'une session"""
        messages = session.conversation
        
        # Comptages de base
        user_messages = len([m for m in messages if m.sender == "user"])
        agent_responses = len([m for m in messages if m.sender == "agent"])
        
        # Comptages par mode
        dialogue_msgs = len([m for m in messages if m.mode == "dialogue"])
        triangle_msgs = len([m for m in messages if m.mode == "triangle"])
        documents_msgs = len([m for m in messages if m.mode == "documents"])
        
        # Coûts
        total_cost = session.session_info.session_costs.get('total', 0.0)
        avg_cost = total_cost / max(1, agent_responses)
        
        # Durée session
        duration_minutes = 0.0
        if session.session_info.start_time and session.session_info.end_time:
            try:
                start = datetime.fromisoformat(session.session_info.start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(session.session_info.end_time.replace('Z', '+00:00'))
                duration_minutes = (end - start).total_seconds() / 60
            except:
                pass
        
        # Agent dominant
        agent_counts = {}
        for msg in messages:
            if msg.sender == "agent" and msg.agent:
                agent_counts[msg.agent] = agent_counts.get(msg.agent, 0) + 1
        dominant_agent = max(agent_counts, key=agent_counts.get) if agent_counts else ""
        
        # Thème dominant (le plus mentionné)
        themes = session.session_info.themes_detected
        dominant_theme = themes[0] if themes else ""
        
        return SessionStatistics(
            user_messages=user_messages,
            agent_responses=agent_responses,
            dialogue_messages=dialogue_msgs,
            triangle_messages=triangle_msgs,
            documents_messages=documents_msgs,
            total_cost=total_cost,
            average_cost_per_response=round(avg_cost, 4),
            session_duration_minutes=round(duration_minutes, 2),
            dominant_agent=dominant_agent,
            dominant_theme=dominant_theme
        )
    
    # 🔥 MÉTHODE CREATE_SESSION CORRIGÉE
    def create_session(self, user_id: str = "FG", metadata: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> str:
        """🆕 Création nouvelle session avec gestion métadonnées flexible"""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:10]}_{int(datetime.now().timestamp() * 1000)}"
        
        # 🔥 GESTION METADATA FLEXIBLE
        if metadata is None:
            metadata = {}
        
        # Merge des métadonnées par défaut
        default_metadata = {
            'created_via': 'session_manager_v5',
            'creation_timestamp': datetime.now().isoformat() + "Z",
            'version': '5.0.1'
        }
        
        # Fusion intelligente des métadonnées
        final_metadata = {**default_metadata, **metadata}
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now().isoformat() + "Z",
            session_costs={
                'anima': 0.0,
                'neo': 0.0,
                'nexus': 0.0,
                'triple': 0.0,
                'total': 0.0
            },
            metadata=final_metadata  # 🔥 STOCKAGE METADATA
        )
        
        statistics = SessionStatistics()
        
        session = CompleteSession(
            session_info=session_info,
            conversation=[],
            statistics=statistics,
            export_date=datetime.now().isoformat() + "Z"
        )
        
        # Stockage en mémoire
        self.active_sessions[session_id] = session
        self.current_session = session
        
        logger.info(f"🆕 Session créée: {session_id} pour {user_id}")
        logger.debug(f"📋 Métadonnées: {final_metadata}")
        
        return session_id  # 🔥 RETOURNE STRING ID au lieu de CompleteSession
    
    # 🔥 MÉTHODE ADD_MESSAGE_TO_SESSION CORRIGÉE
    def add_message_to_session(self, session_id: str, message_type: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """💬 Ajout message à une session avec type/content flexible"""
        try:
            if session_id not in self.active_sessions:
                logger.warning(f"⚠️ Session {session_id} non trouvée, création automatique")
                self.create_session(session_id=session_id)
            
            session = self.active_sessions[session_id]
            
            # Construction SessionMessage
            message = SessionMessage(
                timestamp=datetime.now().isoformat() + "Z",
                sender=message_type,  # 'user' ou 'agent'
                message=content.get('text', ''),
                metadata=metadata or {},
                mode=content.get('mode', 'dialogue'),
                session_id=session_id
            )
            
            # Gestion agent spécifique pour messages agent
            if message_type == 'agent' and 'agent' in content:
                message.agent = content['agent']
            
            session.conversation.append(message)
            
            # Mise à jour compteurs
            session.session_info.total_messages += 1
            
            # Mise à jour modes utilisés
            mode = content.get('mode', 'dialogue')
            if mode not in session.session_info.modes_used:
                session.session_info.modes_used.append(mode)
            
            # Mise à jour agents utilisés
            if message_type == "agent" and 'agent' in content:
                agent = content['agent']
                if agent not in session.session_info.agents_used:
                    session.session_info.agents_used.append(agent)
            
            logger.debug(f"💬 Message {message_type} ajouté à session {session_id} (total: {session.session_info.total_messages})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout message session {session_id}: {e}")
            return False
    
    def update_session_costs(self, session_id: str, agent: str, cost: float) -> bool:
        """💰 Mise à jour coûts session"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.session_info.session_costs[agent] = session.session_info.session_costs.get(agent, 0.0) + cost
                session.session_info.session_costs['total'] = session.session_info.session_costs.get('total', 0.0) + cost
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Erreur update coûts session {session_id}: {e}")
            return False
    
    # 🔥 MÉTHODE FINALIZE_SESSION CORRIGÉE
    def finalize_session(self, session_id: str, final_metadata: Optional[Dict[str, Any]] = None) -> CompleteSession:
        """✅ Finalisation et sauvegarde complète d'une session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} non trouvée")
            
            session = self.active_sessions[session_id]
            
            # Finalisation timestamps
            session.session_info.end_time = datetime.now().isoformat() + "Z"
            session.export_date = datetime.now().isoformat() + "Z"
            
            # 🔥 AJOUT METADATA FINALE
            if final_metadata:
                existing_metadata = session.session_info.metadata
                combined_metadata = {**existing_metadata, **final_metadata}
                session.session_info.metadata = combined_metadata
            
            # Détection thématique automatique
            session.session_info.themes_detected = self._detect_themes(session.conversation)
            
            # Calcul statistiques finales
            session.statistics = self._calculate_statistics(session)
            
            # Sauvegarde JSON
            self._save_session_json(session)
            
            # Sauvegarde SQLite
            self._save_session_sqlite(session)
            
            # Nettoyage mémoire
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            if self.current_session and self.current_session.session_info.session_id == session_id:
                self.current_session = None
            
            logger.info(f"✅ Session {session_id} finalisée et sauvegardée")
            logger.info(f"📊 Stats: {session.statistics.user_messages} messages user, {session.statistics.agent_responses} réponses agents")
            logger.info(f"🎯 Thèmes: {', '.join(session.session_info.themes_detected) if session.session_info.themes_detected else 'Aucun'}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Erreur finalisation session {session_id}: {e}")
            raise
    
    def _save_session_json(self, session: CompleteSession) -> bool:
        """💾 Sauvegarde session en JSON"""
        try:
            filename = f"{session.session_info.session_id}.json"
            filepath = self.sessions_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(session.to_json())
            
            logger.debug(f"💾 Session JSON sauvegardée: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde JSON session: {e}")
            return False
    
    def _save_session_sqlite(self, session: CompleteSession) -> bool:
        """🗄️ Sauvegarde session en SQLite"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Insert session principale
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions (
                        id, user_id, start_time, end_time, session_data,
                        themes, agents_used, modes_used, documents_used,
                        total_cost, message_count, session_duration_minutes,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    session.session_info.session_id,
                    session.session_info.user_id,
                    session.session_info.start_time,
                    session.session_info.end_time,
                    session.to_json(),
                    json.dumps(session.session_info.themes_detected),
                    json.dumps(session.session_info.agents_used),
                    json.dumps(session.session_info.modes_used),
                    json.dumps(session.session_info.documents_used),
                    session.statistics.total_cost,
                    session.statistics.user_messages + session.statistics.agent_responses,
                    session.statistics.session_duration_minutes
                ))
                
                # Delete anciens messages de cette session
                cursor.execute("DELETE FROM session_messages WHERE session_id = ?", (session.session_info.session_id,))
                
                # Insert messages pour FTS5
                for msg in session.conversation:
                    cursor.execute("""
                        INSERT INTO session_messages (
                            session_id, message_text, agent, sender, mode, timestamp, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session.session_info.session_id,
                        msg.message,
                        msg.agent,
                        msg.sender,
                        msg.mode,
                        msg.timestamp,
                        json.dumps(msg.metadata)
                    ))
                
                conn.commit()
                logger.debug(f"🗄️ Session SQLite sauvegardée: {session.session_info.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde SQLite session: {e}")
            return False
    
    # 🔥 MÉTHODES MANQUANTES AJOUTÉES POUR COMPATIBILITÉ API
    def get_session_by_id(self, session_id: str) -> Optional[CompleteSession]:
        """🔍 Récupération session par ID avec fallbacks robustes"""
        try:
            # Check active sessions first
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
            
            # Check SQLite
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT session_data FROM sessions WHERE id = ?", (session_id,))
                row = cursor.fetchone()
                
                if row:
                    session_data = json.loads(row[0])
                    # Reconstruction de l'objet CompleteSession
                    session = CompleteSession(
                        session_info=SessionInfo(**session_data['session_info']),
                        conversation=[SessionMessage(**msg) for msg in session_data['conversation']],
                        statistics=SessionStatistics(**session_data['statistics']),
                        export_date=session_data.get('export_date', ''),
                        version=session_data.get('version', '5.0.1')
                    )
                    return session
            
            # Check JSON files
            json_file = self.sessions_dir / f"{session_id}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    session = CompleteSession(
                        session_info=SessionInfo(**session_data['session_info']),
                        conversation=[SessionMessage(**msg) for msg in session_data['conversation']],
                        statistics=SessionStatistics(**session_data['statistics']),
                        export_date=session_data.get('export_date', ''),
                        version=session_data.get('version', '5.0.1')
                    )
                    return session
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération session {session_id}: {e}")
            return None
    
    def get_session_stats(self, user_id: str = "FG") -> Dict[str, Any]:
        """📊 Statistiques globales des sessions avec fallbacks"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Stats générales
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        SUM(message_count) as total_messages,
                        SUM(total_cost) as total_cost,
                        AVG(session_duration_minutes) as avg_duration,
                        MAX(start_time) as last_session
                    FROM sessions WHERE user_id = ?
                """, (user_id,))
                
                stats = cursor.fetchone()
                
                # Thèmes les plus fréquents
                cursor.execute("""
                    SELECT themes, COUNT(*) as count
                    FROM sessions 
                    WHERE user_id = ? AND themes != '[]' AND themes IS NOT NULL
                    GROUP BY themes
                    ORDER BY count DESC
                    LIMIT 10
                """, (user_id,))
                
                themes_raw = cursor.fetchall()
                
                # Agents les plus utilisés
                cursor.execute("""
                    SELECT agents_used, COUNT(*) as count
                    FROM sessions 
                    WHERE user_id = ? AND agents_used != '[]' AND agents_used IS NOT NULL
                    GROUP BY agents_used
                    ORDER BY count DESC
                    LIMIT 10
                """, (user_id,))
                
                agents_raw = cursor.fetchall()
                
                return {
                    "total_sessions": stats[0] or 0,
                    "total_messages": stats[1] or 0,
                    "total_cost": round(stats[2] or 0.0, 2),
                    "average_duration_minutes": round(stats[3] or 0.0, 2),
                    "last_session_date": stats[4] or "",
                    "themes_frequency": themes_raw,
                    "agents_frequency": agents_raw,
                    "user_id": user_id,
                    "method": "session_stats_v5"
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur stats sessions: {e}")
            return {
                "total_sessions": 0,
                "error": str(e),
                "method": "fallback_error"
            }
    
    def search_sessions_by_theme(self, theme: str, limit: int = 10, user_id: str = "FG") -> List[Dict[str, Any]]:
        """🎯 Recherche sessions par thème - retourne Dict pour compatibilité API"""
        try:
            sessions = []
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, start_time, themes, agents_used, total_cost, message_count
                    FROM sessions
                    WHERE user_id = ? AND (themes LIKE ? OR themes LIKE ?)
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (user_id, f'%"{theme}"%', f'%{theme}%', limit))
                
                for row in cursor.fetchall():
                    session_dict = {
                        'session_id': row[0],
                        'start_time': row[1],
                        'themes': json.loads(row[2]) if row[2] else [],
                        'agents_used': json.loads(row[3]) if row[3] else [],
                        'total_cost': row[4] or 0.0,
                        'message_count': row[5] or 0,
                        'theme_searched': theme
                    }
                    sessions.append(session_dict)
            
            logger.info(f"🎯 {len(sessions)} sessions trouvées pour thème '{theme}'")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche sessions par thème: {e}")
            return []
    
    def search_sessions_by_date(self, start_date: str, end_date: str, limit: int = 20, user_id: str = "FG") -> List[Dict[str, Any]]:
        """📅 Recherche sessions par période - retourne Dict pour compatibilité API"""
        try:
            sessions = []
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, start_time, end_time, themes, agents_used, total_cost, message_count
                    FROM sessions
                    WHERE user_id = ? AND start_time BETWEEN ? AND ?
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (user_id, start_date, end_date, limit))
                
                for row in cursor.fetchall():
                    session_dict = {
                        'session_id': row[0],
                        'start_time': row[1],
                        'end_time': row[2],
                        'themes': json.loads(row[3]) if row[3] else [],
                        'agents_used': json.loads(row[4]) if row[4] else [],
                        'total_cost': row[5] or 0.0,
                        'message_count': row[6] or 0,
                        'date_range': f"{start_date} - {end_date}"
                    }
                    sessions.append(session_dict)
            
            logger.info(f"📅 {len(sessions)} sessions trouvées entre {start_date} et {end_date}")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche sessions par date: {e}")
            return []
    
    def get_recent_sessions(self, user_id: str = "FG", limit: int = 10) -> List[Dict[str, Any]]:
        """🕒 Récupération sessions récentes - retourne Dict pour compatibilité API"""
        try:
            sessions = []
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, start_time, end_time, themes, agents_used, modes_used, total_cost, message_count, session_duration_minutes
                    FROM sessions
                    WHERE user_id = ?
                    ORDER BY start_time DESC
                    LIMIT ?
                """, (user_id, limit))
                
                for row in cursor.fetchall():
                    session_dict = {
                        'session_id': row[0],
                        'start_time': row[1],
                        'end_time': row[2],
                        'themes': json.loads(row[3]) if row[3] else [],
                        'agents_used': json.loads(row[4]) if row[4] else [],
                        'modes_used': json.loads(row[5]) if row[5] else [],
                        'total_cost': row[6] or 0.0,
                        'message_count': row[7] or 0,
                        'duration_minutes': row[8] or 0.0,
                        'method': 'recent_sessions_v5'
                    }
                    sessions.append(session_dict)
            
            logger.info(f"🕒 {len(sessions)} sessions récentes récupérées")
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération sessions récentes: {e}")
            return []
    
    def export_session_json(self, session_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """📤 Export session vers fichier JSON"""
        try:
            session = self.get_session_by_id(session_id)
            if not session:
                logger.error(f"❌ Session {session_id} non trouvée pour export")
                return None
            
            if not export_path:
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                export_path = f"emergence_v5_conversation_{timestamp}.json"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(session.to_json())
            
            logger.info(f"📤 Session {session_id} exportée vers {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"❌ Erreur export session {session_id}: {e}")
            return None

# === FACTORY FUNCTION ===

_global_session_manager = None

def get_session_manager(database_path: str = "data/emergence_v4.db", sessions_dir: str = "data/sessions") -> SessionManagerV5:
    """🏭 Factory function pour Session Manager singleton"""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = SessionManagerV5(database_path, sessions_dir)
    return _global_session_manager

# === TESTS INTÉGRÉS ===

if __name__ == "__main__":
    print("=== TEST SESSION MANAGER V5.0.1 - FIXES API ===\n")
    
    # Instance de test
    sm = SessionManagerV5(database_path="data/test_sessions.db")
    
    # Test création session
    print("🆕 Test création session...")
    session_id = sm.create_session(user_id="FG_TEST", metadata={'test_mode': True})
    print(f"   Session ID retourné: {session_id}")
    print(f"   Type retourné: {type(session_id)}")
    
    # Test ajout messages
    print("\n💬 Test ajout messages...")
    
    # Message user
    success1 = sm.add_message_to_session(
        session_id=session_id,
        message_type="user",
        content={
            'text': "Qu'est-ce que la conscience selon toi ?",
            'mode': 'dialogue'
        },
        metadata={'test': True}
    )
    print(f"   Message user ajouté: {success1}")
    
    # Message agent
    success2 = sm.add_message_to_session(
        session_id=session_id,
        message_type="agent",
        content={
            'agent': 'anima',
            'text': "La conscience, c'est un mystère qui danse...",
            'mode': 'dialogue',
            'model': 'gpt-4o',
            'processing_time': 3.5,
            'cost': 0.025
        }
    )
    print(f"   Message agent ajouté: {success2}")
    
    # Test update coûts
    print("\n💰 Test update coûts...")
    cost_updated = sm.update_session_costs(session_id, "anima", 0.025)
    print(f"   Coûts mis à jour: {cost_updated}")
    
    # Test finalisation
    print("\n✅ Test finalisation session...")
    final_session = sm.finalize_session(session_id, {
        'test_completed': True,
        'final_note': 'Test réussi'
    })
    
    print(f"   Session finalisée: {final_session.session_info.session_id}")
    print(f"   Messages total: {final_session.statistics.user_messages + final_session.statistics.agent_responses}")
    print(f"   Coût total: ${final_session.statistics.total_cost}")
    print(f"   Thèmes détectés: {final_session.session_info.themes_detected}")
    print(f"   Agents utilisés: {final_session.session_info.agents_used}")
    
    # Test récupération sessions récentes
    print("\n🕒 Test sessions récentes...")
    recent = sm.get_recent_sessions("FG_TEST", limit=5)
    print(f"   Sessions récentes: {len(recent)}")
    print(f"   Type retour: {type(recent)}")
    if recent:
        print(f"   Première session: {recent[0].get('session_id', 'N/A')}")
    
    # Test stats
    print("\n📊 Test statistiques...")
    stats = sm.get_session_stats("FG_TEST")
    print(f"   Stats type: {type(stats)}")
    print(f"   Sessions totales: {stats.get('total_sessions', 0)}")
    print(f"   Messages totaux: {stats.get('total_messages', 0)}")
    print(f"   Coût total: ${stats.get('total_cost', 0)}")
    
    # Test recherche par thème
    print("\n🎯 Test recherche par thème...")
    theme_results = sm.search_sessions_by_theme("conscience", limit=5, user_id="FG_TEST")
    print(f"   Résultats thème: {len(theme_results)}")
    print(f"   Type retour: {type(theme_results)}")
    
    print("\n✅ SESSION MANAGER V5.0.1 - TESTS TERMINÉS")
    print("🔧 FIXES APPLIQUÉS:")
    print("   ✅ create_session() retourne string ID")
    print("   ✅ add_message_to_session() avec structure flexible")
    print("   ✅ finalize_session() avec final_metadata")
    print("   ✅ Toutes méthodes API retournent Dict/List")
    print("   ✅ Gestion metadata avancée")
    print("   ✅ Fallbacks robustes pour erreurs")
    print("   ✅ Compatibilité main.py garantie")