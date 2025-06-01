"""
ÉMERGENCE v2 - Core Types Module
Types et structures de données pour l'écosystème ÉMERGENCE
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# ENUMS ET CONSTANTES
# ==========================================

class AgentType(Enum):
    """Types d'agents disponibles"""
    ANIMA = "anima"
    NEO = "neo"
    HYBRID = "hybrid"

class MessageRole(Enum):
    """Rôles dans une conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class DocumentType(Enum):
    """Types de documents supportés"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    JSON = "json"
    CSV = "csv"

class ChunkingStrategy(Enum):
    """Stratégies de découpage"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"

# ==========================================
# STRUCTURES DE DONNÉES PRINCIPALES
# ==========================================

@dataclass
class ConversationEntry:
    """
    Entrée de conversation entre utilisateur et agent
    """
    id: str
    user_input: str
    agent_response: str
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'id': self.id,
            'user_input': self.user_input,
            'agent_response': self.agent_response,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationEntry':
        """Crée depuis un dictionnaire"""
        return cls(
            id=data['id'],
            user_input=data['user_input'],
            agent_response=data['agent_response'],
            agent_name=data['agent_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

@dataclass
class Document:
    """
    Document avec métadonnées et contenu
    """
    id: str
    title: str
    content: str
    source: str
    document_type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'document_type': self.document_type.value,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Crée depuis un dictionnaire"""
        return cls(
            id=data['id'],
            title=data['title'],
            content=data['content'],
            source=data['source'],
            document_type=DocumentType(data['document_type']),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at'])
        )

@dataclass
class ChunkMetadata:
    """
    Métadonnées pour un chunk de document
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_size: int
    strategy: ChunkingStrategy
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_size': self.chunk_size,
            'strategy': self.strategy.value,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Crée depuis un dictionnaire"""
        return cls(
            chunk_id=data['chunk_id'],
            document_id=data['document_id'],
            chunk_index=data['chunk_index'],
            start_char=data['start_char'],
            end_char=data['end_char'],
            chunk_size=data['chunk_size'],
            strategy=ChunkingStrategy(data['strategy']),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )

@dataclass
class DocumentChunk:
    """
    Chunk de document avec contenu et métadonnées
    """
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'embedding': self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Crée depuis un dictionnaire"""
        return cls(
            content=data['content'],
            metadata=ChunkMetadata.from_dict(data['metadata']),
            embedding=data.get('embedding')
        )

# ==========================================
# TYPES POUR RAG ET RECHERCHE
# ==========================================

@dataclass
class SearchResult:
    """
    Résultat de recherche avec score de similarité
    """
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'content': self.content,
            'source': self.source,
            'score': self.score,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }

@dataclass
class RAGContext:
    """
    Contexte RAG pour génération de réponse
    """
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'total_results': self.total_results,
            'search_time': self.search_time,
            'metadata': self.metadata
        }

# ==========================================
# TYPES ADVANCED RAG
# ==========================================

@dataclass
class Entity:
    """
    Entité extraite d'un document
    """
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'text': self.text,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'document_id': self.document_id,
            'metadata': self.metadata
        }

@dataclass
class Relation:
    """
    Relation entre deux entités ou concepts
    """
    source: str
    target: str
    relation_type: str
    confidence: float
    context: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'confidence': self.confidence,
            'context': self.context,
            'document_id': self.document_id,
            'metadata': self.metadata
        }

@dataclass
class Concept:
    """
    Concept extrait avec score d'importance
    """
    text: str
    importance_score: float
    frequency: int
    document_ids: List[str]
    related_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'text': self.text,
            'importance_score': self.importance_score,
            'frequency': self.frequency,
            'document_ids': self.document_ids,
            'related_concepts': self.related_concepts,
            'metadata': self.metadata
        }

# ==========================================
# TYPES POUR MESSAGES ET CHAT
# ==========================================

@dataclass
class ChatMessage:
    """
    Message de chat avec rôle et métadonnées
    """
    role: MessageRole
    content: str
    agent_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'role': self.role.value,
            'content': self.content,
            'agent_name': self.agent_name,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Crée depuis un dictionnaire"""
        return cls(
            role=MessageRole(data['role']),
            content=data['content'],
            agent_name=data.get('agent_name'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )

@dataclass
class Conversation:
    """
    Conversation complète avec historique
    """
    id: str
    title: Optional[str]
    messages: List[ChatMessage]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ChatMessage):
        """Ajoute un message à la conversation"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'id': self.id,
            'title': self.title,
            'messages': [m.to_dict() for m in self.messages],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

# ==========================================
# TYPES POUR DÉBATS AUTONOMES
# ==========================================

@dataclass
class DebateEntry:
    """
    Entrée dans un débat autonome
    """
    round_number: int
    agent_name: str
    content: str
    topic: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'round_number': self.round_number,
            'agent_name': self.agent_name,
            'content': self.content,
            'topic': self.topic,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class DebateSession:
    """
    Session de débat complète
    """
    id: str
    topic: str
    entries: List[DebateEntry]
    max_rounds: int
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, entry: DebateEntry):
        """Ajoute une entrée au débat"""
        self.entries.append(entry)
    
    def is_completed(self) -> bool:
        """Vérifie si le débat est terminé"""
        return len(self.entries) >= self.max_rounds * 2  # 2 agents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'id': self.id,
            'topic': self.topic,
            'entries': [e.to_dict() for e in self.entries],
            'max_rounds': self.max_rounds,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata
        }

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def create_document_from_upload(uploaded_file, content: str) -> Document:
    """
    Crée un Document depuis un fichier uploadé Streamlit
    
    Args:
        uploaded_file: Fichier Streamlit UploadedFile
        content: Contenu extrait du fichier
        
    Returns:
        Document configuré
    """
    import uuid
    
    # Détection du type
    file_ext = uploaded_file.name.split('.')[-1].lower()
    doc_type = DocumentType.TXT  # Défaut
    
    if file_ext == 'pdf':
        doc_type = DocumentType.PDF
    elif file_ext == 'docx':
        doc_type = DocumentType.DOCX
    elif file_ext == 'md':
        doc_type = DocumentType.MD
    elif file_ext == 'json':
        doc_type = DocumentType.JSON
    elif file_ext == 'csv':
        doc_type = DocumentType.CSV
    
    return Document(
        id=str(uuid.uuid4()),
        title=uploaded_file.name,
        content=content,
        source=uploaded_file.name,
        document_type=doc_type,
        metadata={
            'file_size': uploaded_file.size if hasattr(uploaded_file, 'size') else len(content),
            'file_type': uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
        }
    )

def create_conversation_entry(user_input: str, agent_response: str, agent_name: str) -> ConversationEntry:
    """
    Crée une ConversationEntry rapidement
    
    Args:
        user_input: Message utilisateur
        agent_response: Réponse agent
        agent_name: Nom de l'agent
        
    Returns:
        ConversationEntry configurée
    """
    import uuid
    
    return ConversationEntry(
        id=str(uuid.uuid4()),
        user_input=user_input,
        agent_response=agent_response,
        agent_name=agent_name
    )

def create_search_result(content: str, source: str, score: float, **kwargs) -> SearchResult:
    """
    Crée un SearchResult rapidement
    
    Args:
        content: Contenu du résultat
        source: Source du contenu
        score: Score de similarité
        **kwargs: Métadonnées additionnelles
        
    Returns:
        SearchResult configuré
    """
    return SearchResult(
        content=content,
        source=source,
        score=score,
        metadata=kwargs
    )

# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    # Enums
    'AgentType', 'MessageRole', 'DocumentType', 'ChunkingStrategy',
    
    # Structures principales
    'ConversationEntry', 'Document', 'ChunkMetadata', 'DocumentChunk',
    
    # RAG et recherche
    'SearchResult', 'RAGContext',
    
    # Advanced RAG
    'Entity', 'Relation', 'Concept',
    
    # Chat et messages
    'ChatMessage', 'Conversation',
    
    # Débats
    'DebateEntry', 'DebateSession',
    
    # Fonctions utilitaires
    'create_document_from_upload', 'create_conversation_entry', 'create_search_result'
]

logger.info("Module 'core.types' Phase 3A+3B chargé - Types pour Upload Documents + Discussions Autonomes")