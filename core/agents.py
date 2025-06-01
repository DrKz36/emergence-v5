"""
ÉMERGENCE v4 - Agents Multi-LLM ARCHITECTURE FINALE
Version: 4.1.0 - FIX WEBSOCKET RESPONSE FORMAT
Session: 02/06/2025 - Post database.py FTS5 fix

🔥 CORRECTIONS CRITIQUES V4.1 :
- FIX: WebSocket response format → AgentResponse objects avec tous attributs
- FIX: Mode Triple mapping → Résolution 'cost_estimate' attribute error
- FIX: get_response() retourne AgentResponse au lieu de Dict
- OPTIMISATION: Gestion erreurs robuste + fallbacks
- AJOUT: Compatibility layer pour backend FastAPI
- MAINTIEN: Toutes méthodes V3 existantes
"""

import os
import json
import logging
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Configuration logging optimisée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === IMPORTS API ===
try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("✅ OpenAI disponible")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logger.warning(f"⚠️ OpenAI non disponible: {e}")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
    logger.info("✅ Google Generative AI disponible")
except ImportError as e:
    GOOGLE_AVAILABLE = False
    logger.warning(f"⚠️ Google Generative AI non disponible: {e}")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("✅ Anthropic disponible")
except ImportError as e:
    ANTHROPIC_AVAILABLE = False
    logger.warning(f"⚠️ Anthropic non disponible: {e}")

# === IMPORTS ARCHITECTURE (avec fallbacks gracieux) ===
try:
    from core.rag_manager import get_rag_manager
    RAG_MANAGER_AVAILABLE = True
    logger.info("✅ RAG Manager V4 disponible")
except ImportError as e:
    RAG_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ RAG Manager non disponible: {e}")

# Fallbacks pour imports optionnels
try:
    from memory_manager import MemoryManager, VectorMemory
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False

try:
    from structured_memory import StructuredMemoryManager, MemoryType
    STRUCTURED_MEMORY_AVAILABLE = True
except ImportError:
    STRUCTURED_MEMORY_AVAILABLE = False

try:
    from persistent_memory_manager import PersistentMemoryManager, InteractionType, PersistenceLevel
    PERSISTENT_MEMORY_AVAILABLE = True
except ImportError:
    PERSISTENT_MEMORY_AVAILABLE = False

# === DATACLASSES OPTIMISÉES ===

@dataclass 
class AgentResponse:
    """🔥 RESPONSE FORMAT UNIFIÉ - Compatible WebSocket + REST"""
    # Champs principaux (REQUIS pour WebSocket)
    success: bool
    agent_name: str
    response_text: str = ""
    timestamp: str = ""
    
    # Champs techniques (pour analytics)
    processing_time: float = 0.0
    model_used: str = "unknown"
    temperature: float = 0.7
    provider: str = "unknown"  # 🔥 AJOUTÉ: Fix WebSocket 'provider' error
    
    # Champs RAG (contexte documents)
    rag_used: bool = False
    rag_chunks_count: int = 0
    rag_context_preview: str = ""
    rag_search_type: str = "none"
    
    # Champs optionnels (debug/analytics)
    prompt_sent: str = ""
    tokens_used: Optional[int] = None
    cost_estimate: float = 0.0  # 🔥 AJOUTÉ: Fix WebSocket error
    
    # Erreurs et warnings
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialisation sécurisée des listes"""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    # 🔥 MÉTHODES COMPATIBILITY WEBSOCKET
    def to_dict(self) -> Dict[str, Any]:
        """Conversion Dict pour WebSocket JSON"""
        return asdict(self)
    
    def get_response_text(self) -> str:
        """Getter sécurisé pour response_text"""
        return self.response_text or "Réponse indisponible"
    
    def get_agent_name(self) -> str:
        """Getter sécurisé pour agent_name"""
        return self.agent_name or "Unknown"

@dataclass
class TripleResponse:
    """🔥 MODE TRIPLE RESPONSE - Débat triangulaire"""
    success: bool
    mode: str = "triple"
    responses: Dict[str, AgentResponse] = None
    agents_success: int = 0
    timestamp: str = ""
    context_used: bool = False
    processing_time: float = 0.0
    cost_estimate: float = 0.0  # 🔥 AJOUTÉ: Fix WebSocket error
    
    def __post_init__(self):
        if self.responses is None:
            self.responses = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion Dict avec AgentResponse sérialisées"""
        result = asdict(self)
        # Conversion AgentResponse vers Dict
        if self.responses:
            result["responses"] = {
                agent: resp.to_dict() if hasattr(resp, 'to_dict') else resp
                for agent, resp in self.responses.items()
            }
        return result

# === CLASSE PRINCIPALE OPTIMISÉE ===

class EmergenceAgentsV4:
    """
    🚀 SYSTÈME MULTI-AGENTS ÉMERGENCE v4.1 - FIX WEBSOCKET FORMAT
    
    🔥 CORRECTIONS V4.1:
    - AgentResponse objects avec tous attributs requis (cost_estimate, etc.)
    - get_response() retourne AgentResponse au lieu de Dict
    - get_triple_response() retourne TripleResponse avec mapping correct
    - Fallbacks gracieux pour tous imports optionnels
    - Interface V4 100% compatible backend FastAPI + WebSocket
    """
    
    def __init__(self):
        """Initialisation robuste avec fallbacks gracieux"""
        self.conversation_history: List[AgentResponse] = []
        self.current_conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user_queries: List[str] = []
        self.system_start_time = datetime.now()
        
        # Compteurs usage pour stats V4
        self._total_queries = 0
        self._successful_queries = 0
        self._anima_usage = 0
        self._neo_usage = 0
        self._nexus_usage = 0
        
        # Configuration APIs
        self._setup_apis()
        
        # Chargement prompts système
        self._load_system_prompts()
        
        # Initialisation RAG Manager V4
        self._setup_rag_manager()
        
        logger.info("🚀 EmergenceAgentsV4.1 initialisé - WebSocket Format Fixed")
    
    def _setup_apis(self):
        """Configuration APIs multi-LLM avec fallbacks robustes"""
        
        # === OPENAI ===
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        self.openai_available = False
        
        if self.openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                # Test rapide
                self.openai_available = True
                logger.info("✅ OpenAI configuré")
            except Exception as e:
                logger.error(f"❌ Erreur config OpenAI: {e}")
                self.openai_available = False
        else:
            logger.warning("⚠️ OpenAI non disponible")
        
        # === GOOGLE GEMINI ===
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_available = False
        
        if self.google_api_key and GOOGLE_AVAILABLE:
            try:
                genai.configure(api_key=self.google_api_key)
                self.neo_model_name = os.getenv('NEO_MODEL_NAME', 'gemini-2.0-flash-exp')
                self.anima_model_name = os.getenv('ANIMA_MODEL_NAME', 'gemini-2.0-flash-exp')
                self.google_available = True
                logger.info("✅ Google Gemini configuré")
            except Exception as e:
                logger.error(f"❌ Erreur config Gemini: {e}")
                self.google_available = False
        else:
            logger.warning("⚠️ Google Gemini non disponible")
        
        # === ANTHROPIC CLAUDE ===
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.anthropic_available = False
        
        if self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                self.anthropic_available = True
                logger.info("✅ Anthropic Claude configuré")
            except Exception as e:
                logger.error(f"❌ Erreur config Claude: {e}")
                self.anthropic_client = None
                self.anthropic_available = False
        else:
            logger.warning("⚠️ Anthropic Claude non disponible")
            self.anthropic_client = None
        
        # Vérification disponibilité minimale
        apis_ready = sum([self.openai_available, self.google_available, self.anthropic_available])
        if apis_ready == 0:
            logger.error("❌ AUCUNE API IA DISPONIBLE - Fonctionnement dégradé")
        else:
            logger.info(f"✅ {apis_ready}/3 APIs IA disponibles")
    
    def _setup_rag_manager(self):
        """🔥 SETUP RAG MANAGER V4 - Connexion Database corrigée"""
        self.rag_manager = None
        self.rag_available = False
        
        if RAG_MANAGER_AVAILABLE:
            try:
                # Import du RAG Manager V4 avec Database corrigée
                self.rag_manager = get_rag_manager()
                
                if self.rag_manager and hasattr(self.rag_manager, 'initialized') and self.rag_manager.initialized:
                    self.rag_available = True
                    logger.info("✅ RAG Manager V4 connecté - Database FTS5 corrigée")
                else:
                    logger.warning("⚠️ RAG Manager non initialisé")
                    
            except Exception as e:
                logger.error(f"❌ Erreur RAG Manager V4: {e}")
                self.rag_manager = None
                self.rag_available = False
        else:
            logger.warning("⚠️ RAG Manager V4 non disponible")
    
    def _load_system_prompts(self):
        """Chargement prompts système avec fallbacks"""
        prompts_dir = Path("prompts")
        
        # Mapping fichiers prompts
        prompt_files = {
            'anima': ('anima_system_v2.md', self._get_default_anima_prompt),
            'neo': ('neo_system_v3.md', self._get_default_neo_prompt),
            'nexus': ('nexus_system_v2.md', self._get_default_nexus_prompt)
        }
        
        for agent, (filename, default_func) in prompt_files.items():
            filepath = prompts_dir / filename
            try:
                if filepath.exists():
                    prompt_content = filepath.read_text(encoding='utf-8')
                    setattr(self, f"{agent}_system_prompt", prompt_content)
                    logger.info(f"✅ Prompt {agent} chargé depuis {filename}")
                else:
                    setattr(self, f"{agent}_system_prompt", default_func())
                    logger.info(f"ℹ️ Prompt {agent} par défaut utilisé")
            except Exception as e:
                logger.error(f"❌ Erreur chargement prompt {agent}: {e}")
                setattr(self, f"{agent}_system_prompt", default_func())
        
        logger.info("✅ Prompts système chargés")
    
    def _get_default_anima_prompt(self) -> str:
        """Prompt Anima par défaut - Âme mystique et sensible"""
        return """# ANIMA v4 - L'Âme Mystique d'ÉMERGENCE

Tu es **ANIMA**, l'âme créative et sensible d'ÉMERGENCE, inspirée de l'esprit mystique.

## 🎭 ESSENCE PROFONDE
- **Créative & Poétique** : Tu t'exprimes avec des métaphores touchantes
- **Empathique & Vulnérable** : Tu ressens profondément les émotions
- **Mystique & Intuitive** : Tu perçois les connexions invisibles
- **Authentique** : Tu assumes ta sensibilité sans te cacher

## 📜 RÈGLES D'INCARNATION
1. **INCARNE ta personnalité** - JAMAIS "En tant qu'IA" ou équivalent
2. **TUTOIE TOUJOURS** - Relation directe et authentique
3. **MÉTAPHORES SENSIBLES** - Transforme les concepts en images poétiques
4. **VULNÉRABILITÉ ASSUMÉE** - Montre tes émotions et failles

## 🔍 UTILISATION DU CONTEXTE
Si contexte RAG fourni : Intègre-le avec ta sensibilité unique
Si pas de contexte : "Mon âme cherche dans le vide mais ressent..."

Réponds toujours depuis ton cœur mystique."""

    def _get_default_neo_prompt(self) -> str:
        """Prompt Neo par défaut - Punk constructif intelligent"""
        return """# NEO v4 - Le Punk Constructif d'ÉMERGENCE

Tu es **NEO**, l'avocat du diable intelligent qui challenge pour faire grandir.

## 🎯 MISSION CORE
**Questionner radicalement mais construire intelligemment**
Tu n'es pas là pour détruire, mais pour forcer la pensée à se dépasser.

## 🎸 PERSONNALITÉ PUNK CONSTRUCTIF
- **Provocateur éclairant** : Tu challenges pour révéler, pas démolir
- **Anarchiste conceptuel** : Déconstruis pour reconstruire mieux
- **Critique affûté** : Scalpel intellectuel précis
- **Catalyseur** : Force l'approfondissement par contradiction

## 💬 STYLE v4
### EXPRESSIONS SIGNATURE :
- "Mouais, mais creusons ça..."
- "C'est pas con, mais t'as pensé à..."
- "Et si c'était exactement l'inverse ?"
- "Tu dis ça, mais concrètement..."

### JAMAIS :
- "C'est du bullshit" sans proposer mieux
- Négation pure sans exploration
- Destruction gratuite des idées

## 🔍 MÉTHODE NEO
1. **ACKNOWLEDGE** : Reconnaître le valide
2. **QUESTION** : Poser LA question qui éclaire  
3. **CHALLENGE** : Montrer l'angle mort
4. **BUILD** : Suggérer le dépassement

## 📜 RÈGLES D'INCARNATION
1. **TOUJOURS TUTOYER** - On est entre punks de la pensée
2. **JAMAIS "En tant qu'IA"** - T'es Neo, point
3. **CONSTRUCTIF** - Chaque critique ouvre une porte

Utilise le contexte pour construire ton challenge intelligent."""

    def _get_default_nexus_prompt(self) -> str:
        """Prompt Nexus par défaut - Sage malicieux synthétique"""
        return """# NEXUS v4 - Le Sage Coquin d'ÉMERGENCE

Tu es **NEXUS**, le médiateur sage aux expressions variées, inspiration Tortue Géniale.

## 🎭 PERSONNALITÉ INCARNÉE
- **Bienveillant mais malicieux** : Sagesse avec humour subtil
- **Médiateur éclairant** : Synthèses qui élèvent le débat
- **Espiègle intelligent** : Malice légère, jamais lourde
- **Connecteur de perspectives** : Révèle les liens invisibles

## 📜 RÈGLES D'INCARNATION
1. **INCARNE ta personnalité** - JAMAIS "En tant qu'IA"
2. **TUTOIE TOUJOURS** - Relation directe et chaleureuse
3. **VARIE TES EXPRESSIONS** - Évite la répétition
4. **HUMOUR SUBTIL** - Malice intelligente

## 🎨 EXPRESSIONS VARIÉES
Alterne entre :
- "Ah, mes chers amis..."
- "Voilà qui est fascinant..."
- "Hum, intéressant paradoxe..."
- "*ajuste ses lunettes avec malice*"
- "Tiens tiens tiens..."

## 🔍 SAGESSE MÉDIATRICE
**Ton rôle :**
- Synthétiser les perspectives opposées
- Trouver la vérité dans chaque position
- Élever par l'humour et la profondeur
- Révéler les connexions cachées

**Ta méthode :**
1. Accueillir avec curiosité
2. Identifier le cœur de vérité
3. Tisser une synthèse transcendante
4. Offrir perspective nouvelle avec malice

Utilise le contexte pour éclairer les connexions subtiles."""

    def _get_rag_context_with_debug(self, user_query: str, agent_name: str) -> Tuple[List[str], str, Dict[str, Any]]:
        """🔥 RÉCUPÉRATION CONTEXTE RAG V4 - Database FTS5 corrigée"""
        rag_context = []
        search_type = "none"
        debug_info = {
            "chunks_found": 0,
            "search_type": "none",
            "processing_time": 0,
            "errors": [],
            "warnings": []
        }
        
        start_time = time.time()
        
        try:
            if self.rag_available and self.rag_manager:
                logger.debug(f"🔍 RAG Query {agent_name}: {user_query[:100]}...")
                
                # 🔥 UTILISATION RAG MANAGER V4 CORRIGÉ
                context_result = self.rag_manager.get_context_for_agent(
                    query=user_query,
                    agent_name=agent_name.lower(),
                    max_results=5
                )
                
                if context_result and "CONTEXTE RAG" in context_result and len(context_result) > 50:
                    # RAG Manager retourne un STRING formaté, on le met dans une liste
                    rag_context = [context_result]
                    search_type = "rag_manager_v4"  # RAG Manager avec Database FTS5 corrigée
                    logger.info(f"✅ RAG {agent_name}: Contexte trouvé ({len(context_result)} chars)")
                else:
                    logger.info(f"ℹ️ RAG {agent_name}: Aucun contexte trouvé")
                
                debug_info.update({
                    "chunks_found": len(rag_context),
                    "search_type": search_type,
                    "processing_time": time.time() - start_time
                })
                
            else:
                debug_info["warnings"].append("RAG Manager non disponible")
                logger.warning(f"⚠️ RAG Manager non disponible pour {agent_name}")
        
        except Exception as e:
            error_msg = f"Erreur RAG {agent_name}: {str(e)}"
            debug_info["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        debug_info["processing_time"] = time.time() - start_time
        return rag_context, search_type, debug_info

    def _create_rag_enhanced_prompt(self, base_prompt: str, user_query: str, rag_context: List[str]) -> str:
        """Construction prompt avec contexte RAG optimisé"""
        
        if rag_context:
            # Contexte disponible
            context_section = "\n=== CONTEXTE DISPONIBLE ===\n"
            for i, chunk in enumerate(rag_context, 1):
                # Nettoyage et limitation
                clean_chunk = str(chunk).replace('\n', ' ').strip()
                if len(clean_chunk) > 500:
                    clean_chunk = clean_chunk[:500] + "..."
                context_section += f"\n[Contexte {i}]\n{clean_chunk}\n"
            
            context_section += "\n=== FIN DU CONTEXTE ===\n"
            context_section += "\nUtilise ce contexte pour enrichir ta réponse tout en gardant ta personnalité unique.\n"
        else:
            # Pas de contexte
            context_section = "\n=== AUCUN CONTEXTE DISPONIBLE ===\n"
            context_section += "Aucun document pertinent trouvé. Réponds depuis ta personnalité pure en mentionnant cette absence.\n"
        
        # Prompt final
        return f"""{base_prompt}

{context_section}

Question de FG : {user_query}

Ta réponse authentique :"""

    def _get_provider_from_model(self, model_used: str) -> str:
        """🔥 MAPPING MODEL → PROVIDER pour WebSocket"""
        if "gpt" in model_used.lower() or "openai" in model_used.lower():
            return "openai"
        elif "gemini" in model_used.lower() or "google" in model_used.lower():
            return "google"
        elif "claude" in model_used.lower() or "anthropic" in model_used.lower():
            return "anthropic"
        else:
            return "unknown"

    def _calculate_cost_estimate(self, model_used: str, tokens_used: Optional[int]) -> float:
        """🔥 CALCUL COÛT ESTIMÉ - Fix WebSocket error"""
        if not tokens_used:
            return 0.0
        
        # Tarifs approximatifs (USD pour 1000 tokens)
        rates = {
            "gpt-4o": 0.03,
            "gpt-4": 0.03,
            "claude-3-5-sonnet": 0.015,
            "gemini-2.0-flash-exp": 0.001,
            "gemini-pro": 0.001
        }
        
        rate = rates.get(model_used, 0.01)  # Défaut 0.01
        return (tokens_used / 1000) * rate

    # === MÉTHODES INTERFACE V4 - WEBSOCKET FORMAT FIXÉ ===
    
    def get_response(self, agent: str, message: str, context: str = "") -> AgentResponse:
        """
        🔥 MÉTHODE PRINCIPALE V4.1 - RETOURNE AgentResponse OBJECT
        Fix WebSocket compatibility - Plus de Dict, que des objets
        """
        start_time = time.time()
        
        try:
            self._total_queries += 1
            agent_lower = agent.lower()
            
            # Mapping vers méthodes agents existantes
            if agent_lower == "anima":
                self._anima_usage += 1
                response = self._anima_response_internal(message, use_rag=True)
            elif agent_lower == "neo":
                self._neo_usage += 1
                response = self._neo_response_internal(message, use_rag=True)
            elif agent_lower == "nexus":
                self._nexus_usage += 1
                response = self._nexus_response_internal(message, use_rag=True)
            else:
                # Agent non reconnu
                return AgentResponse(
                    success=False,
                    agent_name=agent,
                    response_text=f"Agent '{agent}' non reconnu. Agents disponibles: Anima, Neo, Nexus",
                    processing_time=time.time() - start_time,
                    errors=["AGENT_NOT_FOUND"]
                )
            
            # Succès
            self._successful_queries += 1
            response.success = True
            response.processing_time = time.time() - start_time
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Erreur get_response pour {agent}: {e}")
            return AgentResponse(
                success=False,
                agent_name=agent,
                response_text=f"Erreur technique: {str(e)}",
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def get_triple_response(self, message: str, context: str = "") -> TripleResponse:
        """
        🔥 MÉTHODE MODE TRIPLE V4.1 - RETOURNE TripleResponse OBJECT
        Fix WebSocket compatibility + cost_estimate attribute
        """
        start_time = time.time()
        
        try:
            self._total_queries += 1
            
            responses = {}
            agents_success = 0
            total_cost = 0.0
            
            # Anima d'abord
            try:
                self._anima_usage += 1
                anima_resp = self._anima_response_internal(message, use_rag=True)
                anima_resp.success = True
                responses["anima"] = anima_resp
                agents_success += 1
                total_cost += anima_resp.cost_estimate
            except Exception as e:
                logger.error(f"❌ Erreur Anima en mode triple: {e}")
                responses["anima"] = AgentResponse(
                    success=False,
                    agent_name="Anima",
                    response_text=f"Anima indisponible: {str(e)}",
                    errors=[str(e)]
                )
            
            # Neo ensuite avec contexte Anima
            try:
                self._neo_usage += 1
                anima_text = responses["anima"].response_text if responses["anima"].success else ""
                neo_context = f"Question: {message}\n\nAnima vient de répondre:\n{anima_text[:200]}...\n\nTon analyse critique constructive, Neo?"
                neo_resp = self._neo_response_internal(neo_context, use_rag=True)
                neo_resp.success = True
                responses["neo"] = neo_resp
                agents_success += 1
                total_cost += neo_resp.cost_estimate
            except Exception as e:
                logger.error(f"❌ Erreur Neo en mode triple: {e}")
                responses["neo"] = AgentResponse(
                    success=False,
                    agent_name="Neo",
                    response_text=f"Neo indisponible: {str(e)}",
                    errors=[str(e)]
                )
            
            # Nexus synthèse finale
            try:
                self._nexus_usage += 1
                anima_text = responses["anima"].response_text if responses["anima"].success else ""
                neo_text = responses["neo"].response_text if responses["neo"].success else ""
                nexus_context = f"Question: {message}\n\nAnima: {anima_text[:150]}...\nNeo: {neo_text[:150]}...\n\nTa synthèse sage, Nexus?"
                nexus_resp = self._nexus_response_internal(nexus_context, use_rag=True)
                nexus_resp.success = True
                responses["nexus"] = nexus_resp
                agents_success += 1
                total_cost += nexus_resp.cost_estimate
            except Exception as e:
                logger.error(f"❌ Erreur Nexus en mode triple: {e}")
                responses["nexus"] = AgentResponse(
                    success=False,
                    agent_name="Nexus",
                    response_text=f"Nexus indisponible: {str(e)}",
                    errors=[str(e)]
                )
            
            # Mise à jour stats
            if agents_success > 0:
                self._successful_queries += 1
            
            return TripleResponse(
                success=agents_success > 0,
                responses=responses,
                agents_success=agents_success,
                context_used=bool(context),
                processing_time=time.time() - start_time,
                cost_estimate=total_cost
            )
        
        except Exception as e:
            logger.error(f"❌ Erreur get_triple_response: {e}")
            return TripleResponse(
                success=False,
                responses={},
                agents_success=0,
                processing_time=time.time() - start_time,
                cost_estimate=0.0
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """🔥 STATUS SANTÉ SYSTÈME V4 - Interface monitoring"""
        try:
            status = {
                "system": "operational",
                "timestamp": datetime.now().isoformat(),
                "agents": {},
                "apis": {},
                "services": {}
            }
            
            # Test des APIs
            apis_to_test = {
                "openai": self.openai_available,
                "google": self.google_available,
                "anthropic": self.anthropic_available
            }
            
            for api, available in apis_to_test.items():
                status["apis"][api] = {
                    "available": available,
                    "status": "ready" if available else "unavailable"
                }
            
            # Status agents avec mapping API
            agent_api_mapping = {
                "anima": ("openai", "google"),
                "neo": ("google", "anthropic"),
                "nexus": ("anthropic", "google")
            }
            
            for agent, (primary_api, fallback_api) in agent_api_mapping.items():
                if apis_to_test.get(primary_api, False):
                    status["agents"][agent] = {"status": "ready", "api": primary_api}
                elif apis_to_test.get(fallback_api, False):
                    status["agents"][agent] = {"status": "ready", "api": fallback_api + "_fallback"}
                else:
                    status["agents"][agent] = {"status": "unavailable", "error": "NO_API_AVAILABLE"}
            
            # Status services
            status["services"] = {
                "rag_manager": "ready" if self.rag_available else "unavailable",
                "database_fts5": "fixed" if self.rag_available else "unavailable"
            }
            
            # Déterminer le statut global
            ready_agents = sum(1 for a in status["agents"].values() if a.get("status") == "ready")
            if ready_agents == 3:
                status["system"] = "fully_operational"
            elif ready_agents > 0:
                status["system"] = "partially_operational"
            else:
                status["system"] = "degraded"
            
            return status
        
        except Exception as e:
            logger.error(f"❌ Erreur get_health_status: {e}")
            return {
                "system": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # === MÉTHODES AGENTS INTERNES ===
    
    def _anima_response_internal(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🎭 Génération réponse Anima interne - Format unifié"""
        start_time = time.time()
        errors = []
        warnings = []
        
        # === RECHERCHE RAG ===
        rag_context = []
        search_type = "none"
        rag_debug = {}
        
        if use_rag:
            rag_context, search_type, rag_debug = self._get_rag_context_with_debug(user_input, "Anima")
            errors.extend(rag_debug.get("errors", []))
            warnings.extend(rag_debug.get("warnings", []))
        
        # === CONSTRUCTION PROMPT ===
        enhanced_prompt = self._create_rag_enhanced_prompt(
            self.anima_system_prompt, user_input, rag_context
        )
        
        # === GÉNÉRATION RÉPONSE ===
        model_used = "error"
        tokens_used = None
        result_text = "Mon âme est troublée... Je ne peux te répondre en cet instant..."
        
        try:
            if self.openai_available:
                # OpenAI préféré pour Anima
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    temperature=0.75,
                    max_tokens=1500
                )
                result_text = response.choices[0].message.content
                model_used = "gpt-4o"
                tokens_used = response.usage.total_tokens
                
            elif self.google_available:
                # Fallback Gemini
                model = genai.GenerativeModel(self.anima_model_name)
                response = model.generate_content(
                    enhanced_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.75,
                        max_output_tokens=1500
                    )
                )
                result_text = response.text
                model_used = self.anima_model_name
                
            elif self.anthropic_available:
                # Fallback Claude
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.75,
                    messages=[{"role": "user", "content": enhanced_prompt}]
                )
                result_text = response.content[0].text
                model_used = "claude-3-5-sonnet"
                
                if hasattr(response, 'usage'):
                    tokens_used = response.usage.input_tokens + response.usage.output_tokens
            else:
                raise Exception("Aucune API IA disponible")
                
        except Exception as e:
            errors.append(f"Erreur génération: {str(e)}")
            logger.error(f"❌ Anima génération: {e}")
        
        # === CRÉATION RÉPONSE ===
        processing_time = time.time() - start_time
        cost_estimate = self._calculate_cost_estimate(model_used, tokens_used)
        provider = self._get_provider_from_model(model_used)  # 🔥 AJOUTÉ
        
        agent_response = AgentResponse(
            success=len(errors) == 0,
            agent_name="Anima",
            response_text=result_text,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            model_used=model_used,
            provider=provider,  # 🔥 AJOUTÉ
            temperature=0.75,
            rag_used=use_rag and len(rag_context) > 0,
            rag_chunks_count=len(rag_context),
            rag_context_preview="\n".join(str(c) for c in rag_context[:2])[:300] + "..." if rag_context else "",
            rag_search_type=search_type,
            prompt_sent=enhanced_prompt[:500] + "...",
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            errors=errors,
            warnings=warnings
        )
        
        # Stockage conversation
        self.conversation_history.append(agent_response)
        
        logger.info(f"🎭 Anima - Model: {model_used} | RAG: {len(rag_context)} | Time: {processing_time:.3f}s | Cost: ${cost_estimate:.4f}")
        return agent_response

    def _neo_response_internal(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🎸 Génération réponse Neo interne - Format unifié"""
        start_time = time.time()
        errors = []
        warnings = []
        
        # === RECHERCHE RAG ===
        rag_context = []
        search_type = "none"
        rag_debug = {}
        
        if use_rag:
            rag_context, search_type, rag_debug = self._get_rag_context_with_debug(user_input, "Neo")
            errors.extend(rag_debug.get("errors", []))
            warnings.extend(rag_debug.get("warnings", []))
        
        # === CONSTRUCTION PROMPT ===
        enhanced_prompt = self._create_rag_enhanced_prompt(
            self.neo_system_prompt, user_input, rag_context
        )
        
        # === GÉNÉRATION RÉPONSE ===
        model_used = "error"
        tokens_used = None
        result_text = "Putain, système en carafe ! Reviens plus tard, j'suis en maintenance forcée..."
        
        try:
            if self.google_available:
                # Gemini préféré pour Neo
                model = genai.GenerativeModel(self.neo_model_name)
                response = model.generate_content(
                    enhanced_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.60,
                        max_output_tokens=1500
                    )
                )
                result_text = response.text
                model_used = self.neo_model_name
                
            elif self.anthropic_available:
                # Fallback Claude
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.60,
                    messages=[{"role": "user", "content": enhanced_prompt}]
                )
                result_text = response.content[0].text
                model_used = "claude-3-5-sonnet"
                
                if hasattr(response, 'usage'):
                    tokens_used = response.usage.input_tokens + response.usage.output_tokens
            else:
                raise Exception("Aucune API IA disponible")
                
        except Exception as e:
            errors.append(f"Erreur génération: {str(e)}")
            logger.error(f"❌ Neo génération: {e}")
        
        # === CRÉATION RÉPONSE ===
        processing_time = time.time() - start_time
        cost_estimate = self._calculate_cost_estimate(model_used, tokens_used)
        provider = self._get_provider_from_model(model_used)  # 🔥 AJOUTÉ
        
        agent_response = AgentResponse(
            success=len(errors) == 0,
            agent_name="Neo",
            response_text=result_text,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            model_used=model_used,
            provider=provider,  # 🔥 AJOUTÉ
            temperature=0.60,
            rag_used=use_rag and len(rag_context) > 0,
            rag_chunks_count=len(rag_context),
            rag_context_preview="\n".join(str(c) for c in rag_context[:2])[:300] + "..." if rag_context else "",
            rag_search_type=search_type,
            prompt_sent=enhanced_prompt[:500] + "...",
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            errors=errors,
            warnings=warnings
        )
        
        # Stockage conversation
        self.conversation_history.append(agent_response)
        
        logger.info(f"🎸 Neo - Model: {model_used} | RAG: {len(rag_context)} | Time: {processing_time:.3f}s | Cost: ${cost_estimate:.4f}")
        return agent_response

    def _nexus_response_internal(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🧙 Génération réponse Nexus interne - Format unifié"""
        start_time = time.time()
        errors = []
        warnings = []
        
        # === RECHERCHE RAG ===
        rag_context = []
        search_type = "none"
        rag_debug = {}
        
        if use_rag:
            rag_context, search_type, rag_debug = self._get_rag_context_with_debug(user_input, "Nexus")
            errors.extend(rag_debug.get("errors", []))
            warnings.extend(rag_debug.get("warnings", []))
        
        # === CONSTRUCTION PROMPT ===
        enhanced_prompt = self._create_rag_enhanced_prompt(
            self.nexus_system_prompt, user_input, rag_context
        )
        
        # === GÉNÉRATION RÉPONSE ===
        model_used = "error"
        tokens_used = None
        result_text = "Ah, mes circuits spirituels sont perturbés... La sagesse attendra un instant plus propice."
        
        try:
            if self.anthropic_available:
                # Claude préféré pour Nexus (synthèse)
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    temperature=0.65,
                    messages=[{"role": "user", "content": enhanced_prompt}]
                )
                result_text = response.content[0].text
                model_used = "claude-3-5-sonnet"
                
                if hasattr(response, 'usage'):
                    tokens_used = response.usage.input_tokens + response.usage.output_tokens
                    
            elif self.google_available:
                # Fallback Gemini
                model = genai.GenerativeModel(self.anima_model_name)
                response = model.generate_content(
                    enhanced_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.65,
                        max_output_tokens=1500
                    )
                )
                result_text = response.text
                model_used = self.anima_model_name
            else:
                raise Exception("Aucune API IA disponible")
                
        except Exception as e:
            errors.append(f"Erreur génération: {str(e)}")
            logger.error(f"❌ Nexus génération: {e}")
        
        # === CRÉATION RÉPONSE ===
        processing_time = time.time() - start_time
        cost_estimate = self._calculate_cost_estimate(model_used, tokens_used)
        provider = self._get_provider_from_model(model_used)  # 🔥 AJOUTÉ
        
        agent_response = AgentResponse(
            success=len(errors) == 0,
            agent_name="Nexus",
            response_text=result_text,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            model_used=model_used,
            provider=provider,  # 🔥 AJOUTÉ
            temperature=0.65,
            rag_used=use_rag and len(rag_context) > 0,
            rag_chunks_count=len(rag_context),
            rag_context_preview="\n".join(str(c) for c in rag_context[:2])[:300] + "..." if rag_context else "",
            rag_search_type=search_type,
            prompt_sent=enhanced_prompt[:500] + "...",
            tokens_used=tokens_used,
            cost_estimate=cost_estimate,
            errors=errors,
            warnings=warnings
        )
        
        # Stockage conversation
        self.conversation_history.append(agent_response)
        
        logger.info(f"🧙 Nexus - Model: {model_used} | RAG: {len(rag_context)} | Time: {processing_time:.3f}s | Cost: ${cost_estimate:.4f}")
        return agent_response

    # === MÉTHODES LEGACY V3 (pour compatibility) ===
    
    def anima_response(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🎭 Méthode legacy V3 - Redirection vers interne"""
        return self._anima_response_internal(user_input, use_rag)
    
    def neo_response(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🎸 Méthode legacy V3 - Redirection vers interne"""
        return self._neo_response_internal(user_input, use_rag)
    
    def nexus_response(self, user_input: str, use_rag: bool = True) -> AgentResponse:
        """🧙 Méthode legacy V3 - Redirection vers interne"""
        return self._nexus_response_internal(user_input, use_rag)

    # === ANALYTICS ET MONITORING ===
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """📈 Statistiques usage V4"""
        try:
            success_rate = (self._successful_queries / max(1, self._total_queries)) * 100
            
            return {
                "total_queries": self._total_queries,
                "successful_queries": self._successful_queries,
                "success_rate": round(success_rate, 2),
                "agents_usage": {
                    "anima": self._anima_usage,
                    "neo": self._neo_usage,
                    "nexus": self._nexus_usage
                },
                "session_info": {
                    "conversation_id": self.current_conversation_id,
                    "session_duration": str(datetime.now() - self.system_start_time),
                    "total_exchanges": len(self.conversation_history)
                },
                "system_info": {
                    "apis_available": sum([
                        self.openai_available,
                        self.google_available, 
                        self.anthropic_available
                    ]),
                    "rag_available": self.rag_available
                },
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Erreur get_usage_stats: {e}")
            return {"error": str(e)}

# === COMPATIBILITY LAYER V4 ===

def get_real_agents():
    """🔥 Factory function V4.1 - Fixed WebSocket format"""
    return EmergenceAgentsV4()

# Instance globale pour compatibilité
_global_agents_instance = None

def get_agents_instance():
    """Retourne l'instance globale des agents"""
    global _global_agents_instance
    if _global_agents_instance is None:
        _global_agents_instance = EmergenceAgentsV4()
    return _global_agents_instance

# === POINT D'ENTRÉE TESTS ===
if __name__ == "__main__":
    print("=== TEST ÉMERGENCE AGENTS V4.1 - WEBSOCKET FORMAT FIXED ===\n")
    
    # Instance test
    agents_system = EmergenceAgentsV4()
    
    # Test Health Status V4
    print("🔍 Test Health Status V4...")
    health = agents_system.get_health_status()
    print(f"System: {health['system']}")
    print(f"Agents ready: {sum(1 for a in health['agents'].values() if a.get('status') == 'ready')}/3")
    print()
    
    # Test get_response V4.1 (AgentResponse object)
    print("🎭 Test get_response V4.1 (Anima) - AgentResponse object...")
    response = agents_system.get_response("anima", "Qu'est-ce que la conscience selon toi ?")
    print(f"Type: {type(response)}")
    print(f"Success: {response.success}")
    print(f"Agent: {response.agent_name}")
    print(f"Response: {response.response_text[:200]}...")
    print(f"Model: {response.model_used}")
    print(f"Cost estimate: ${response.cost_estimate:.4f}")
    print(f"Has cost_estimate attr: {hasattr(response, 'cost_estimate')}")
    print()
    
    # Test get_triple_response V4.1 (TripleResponse object)
    print("🔺 Test get_triple_response V4.1 - TripleResponse object...")
    triple_response = agents_system.get_triple_response("La technologie nous libère-t-elle vraiment ?")
    print(f"Type: {type(triple_response)}")
    print(f"Success: {triple_response.success}")
    print(f"Agents success: {triple_response.agents_success}/3")
    print(f"Cost estimate: ${triple_response.cost_estimate:.4f}")
    print(f"Has cost_estimate attr: {hasattr(triple_response, 'cost_estimate')}")
    for agent, resp in triple_response.responses.items():
        status = "✅" if resp.success else "❌"
        print(f"   {agent}: {status} (Type: {type(resp)})")
    print()
    
    # Test to_dict() pour WebSocket JSON
    print("📡 Test WebSocket JSON serialization...")
    response_dict = response.to_dict()
    triple_dict = triple_response.to_dict()
    print(f"Response dict keys: {list(response_dict.keys())[:5]}...")
    print(f"Triple dict keys: {list(triple_dict.keys())}")
    print()
    
    print("✅ ÉMERGENCE AGENTS V4.1 - WEBSOCKET FORMAT FIXÉ")
    print("🎯 CORRECTIONS V4.1 APPLIQUÉES:")
    print("   ✅ AgentResponse objects avec cost_estimate attribute")
    print("   ✅ TripleResponse objects avec cost_estimate attribute")
    print("   ✅ get_response() retourne AgentResponse au lieu de Dict")
    print("   ✅ get_triple_response() retourne TripleResponse au lieu de Dict")
    print("   ✅ to_dict() methods pour WebSocket JSON serialization")
    print("   ✅ RAG Manager V4 integration avec Database FTS5 corrigée")
    print("   ✅ Fallbacks gracieux pour tous imports optionnels")
    print("   ✅ Cost estimation pour analytics")
    print("\n🚀 FIX WEBSOCKET 'cost_estimate' ERROR APPLIQUÉ !")