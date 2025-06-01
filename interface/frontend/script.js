/**
 * ÉMERGENCE V4 - Frontend JavaScript
 * Interface Multi-IA avec WebSocket temps réel
 * Anima (OpenAI) + Neo (Gemini) + Nexus (Claude)
 * 🔥 Simple, efficace, PUISSANT !
 */

// ==========================================
// CONFIGURATION GLOBALE
// ==========================================

const CONFIG = {
    wsUrl: `ws://${window.location.host}/ws/`,
    apiUrl: '/api',
    maxMessages: 100,
    autoScroll: true,
    typingDelay: 1500,
    reconnectDelay: 3000,
    maxFileSize: 10 * 1024 * 1024, // 10MB
    supportedFiles: ['.txt', '.md', '.json', '.csv']
};

// État global application
const AppState = {
    selectedAgent: 'anima',
    useRag: true,
    ragChunks: 5,
    messages: [],
    totalCost: 0.0,
    documentsCount: 0,
    systemStatus: null,
    ws: null,
    sessionId: generateSessionId(),
    isConnected: false,
    isTyping: false,
    lastActivity: Date.now()
};

// Descriptions des agents
const AGENTS_INFO = {
    anima: {
        icon: '🎭',
        name: 'Anima',
        provider: 'OpenAI GPT-4.1',
        description: 'Agent créatif inspiré de Simone Weil',
        color: 'var(--anima-color)'
    },
    neo: {
        icon: '🕶️',
        name: 'Neo',
        provider: 'Gemini 2.5 Pro',
        description: 'Punk constructif, avocat du diable intelligent',
        color: 'var(--neo-color)'
    },
    nexus: {
        icon: '🧙',
        name: 'Nexus',
        provider: 'Claude Sonnet 4',
        description: 'Sage médiateur, synthèses équilibrées',
        color: 'var(--nexus-color)'
    },
    triple: {
        icon: '🔥',
        name: 'Mode Triple',
        provider: 'Débat Triangulaire',
        description: 'Les 3 IA débattent ensemble',
        color: 'var(--triple-color)'
    }
};

// ==========================================
// UTILITAIRES
// ==========================================

function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

function formatCost(cost) {
    return `$${cost.toFixed(4)}`;
}

function formatTime(time) {
    return `${time.toFixed(1)}s`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
            <span>${escapeHtml(message)}</span>
        </div>
    `;
    
    document.getElementById('toast-container').appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function updateCharCount() {
    const input = document.getElementById('message-input');
    const counter = document.getElementById('char-count');
    const current = input.value.length;
    const max = input.maxLength;
    
    counter.textContent = `${current}/${max}`;
    counter.style.color = current > max * 0.9 ? '#ef4444' : 'var(--text-muted)';
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// ==========================================
// GESTION WEBSOCKET
// ==========================================

function connectWebSocket() {
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    const wsUrl = CONFIG.wsUrl + AppState.sessionId;
    console.log('🔗 Connexion WebSocket:', wsUrl);
    
    AppState.ws = new WebSocket(wsUrl);
    
    AppState.ws.onopen = () => {
        console.log('✅ WebSocket connecté');
        AppState.isConnected = true;
        updateSystemStatus('🟢 En ligne');
        showToast('Connexion établie avec les agents IA', 'success');
        
        // Demande status initial
        requestSystemStatus();
    };
    
    AppState.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.error('❌ Erreur parsing WebSocket:', error);
        }
    };
    
    AppState.ws.onclose = () => {
        console.log('🔌 WebSocket fermé');
        AppState.isConnected = false;
        updateSystemStatus('🔴 Déconnecté');
        
        // Reconnexion automatique
        setTimeout(connectWebSocket, CONFIG.reconnectDelay);
    };
    
    AppState.ws.onerror = (error) => {
        console.error('❌ Erreur WebSocket:', error);
        showToast('Erreur de connexion', 'error');
    };
}

function sendWebSocketMessage(data) {
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
        AppState.ws.send(JSON.stringify(data));
        AppState.lastActivity = Date.now();
    } else {
        showToast('Connexion perdue, reconnexion...', 'error');
        connectWebSocket();
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'agent_response':
            handleAgentResponse(data);
            break;
            
        case 'typing':
            showTypingIndicator(data.agent);
            break;
            
        case 'triple_start':
            showTripleStart();
            break;
            
        case 'triple_end':
            hideTypingIndicator();
            showToast(`Débat terminé - Coût: ${formatCost(data.total_cost)}`, 'success');
            break;
            
        case 'status':
            updateSystemStatusFromData(data.system_status);
            AppState.totalCost = data.session_cost || 0;
            updateUI();
            break;
            
        case 'error':
            hideTypingIndicator();
            showToast(data.message, 'error');
            break;
            
        default:
            console.log('📦 Message WebSocket non géré:', data);
    }
}

function requestSystemStatus() {
    sendWebSocketMessage({ type: 'status' });
}

// ==========================================
// GESTION MESSAGES CHAT
// ==========================================

function addMessage(content, sender = 'user', agent = null, metadata = {}) {
    const message = {
        id: generateSessionId(),
        content: content,
        sender: sender,
        agent: agent,
        timestamp: Date.now(),
        metadata: metadata
    };
    
    AppState.messages.push(message);
    
    // Limite nombre de messages
    if (AppState.messages.length > CONFIG.maxMessages) {
        AppState.messages = AppState.messages.slice(-CONFIG.maxMessages);
    }
    
    renderMessage(message);
    
    if (CONFIG.autoScroll) {
        scrollToBottom();
    }
}

function renderMessage(message) {
    const messagesContainer = document.getElementById('chat-messages');
    
    // Supprime message de bienvenue si c'est le premier message
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg && AppState.messages.length === 1) {
        welcomeMsg.remove();
    }
    
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.sender}`;
    messageEl.id = `message-${message.id}`;
    
    if (message.agent) {
        messageEl.classList.add(message.agent.toLowerCase());
    }
    
    let headerInfo = '';
    if (message.sender === 'agent' && message.agent) {
        const agentInfo = AGENTS_INFO[message.agent.toLowerCase()] || AGENTS_INFO.anima;
        const meta = message.metadata;
        
        headerInfo = `
            <div class="message-header">
                <span>${agentInfo.icon} ${agentInfo.name}</span>
                ${meta.provider ? `<span>(${meta.provider})</span>` : ''}
                ${meta.processing_time ? `<span>${formatTime(meta.processing_time)}</span>` : ''}
                ${meta.cost_estimate ? `<span>${formatCost(meta.cost_estimate)}</span>` : ''}
                ${meta.rag_chunks_count > 0 ? `<span>RAG: ${meta.rag_chunks_count}</span>` : ''}
            </div>
        `;
    } else if (message.sender === 'user') {
        headerInfo = `
            <div class="message-header">
                <span>👤 FG</span>
                <span>${new Date(message.timestamp).toLocaleTimeString()}</span>
            </div>
        `;
    }
    
    messageEl.innerHTML = `
        ${headerInfo}
        <div class="message-content">
            ${escapeHtml(message.content).replace(/\n/g, '<br>')}
        </div>
        ${message.metadata.cost_estimate ? `
        <div class="message-footer">
            <div class="message-meta">
                <span>Coût: ${formatCost(message.metadata.cost_estimate)}</span>
                <span>Modèle: ${message.metadata.model_used || 'N/A'}</span>
            </div>
        </div>
        ` : ''}
    `;
    
    messagesContainer.appendChild(messageEl);
}

function handleAgentResponse(data) {
    hideTypingIndicator();
    
    // Mise à jour coût total
    if (data.cost_estimate) {
        AppState.totalCost += data.cost_estimate;
        updateUI();
    }
    
    // Ajouter message agent
    addMessage(
        data.response_text,
        'agent',
        data.agent.toLowerCase(),
        {
            processing_time: data.processing_time,
            cost_estimate: data.cost_estimate,
            provider: data.provider,
            rag_chunks_count: data.rag_chunks_count,
            model_used: data.model_used
        }
    );
}

function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    if (!AppState.isConnected) {
        showToast('Connexion WebSocket fermée', 'error');
        return;
    }
    
    // Ajouter message utilisateur
    addMessage(message, 'user');
    
    // Envoi via WebSocket
    if (AppState.selectedAgent === 'triple') {
        sendWebSocketMessage({
            type: 'triple',
            message: message,
            use_rag: AppState.useRag,
            rag_chunks: AppState.ragChunks
        });
    } else {
        sendWebSocketMessage({
            type: 'chat',
            agent: AppState.selectedAgent,
            message: message,
            use_rag: AppState.useRag,
            rag_chunks: AppState.ragChunks
        });
    }
    
    // Reset input
    input.value = '';
    input.style.height = 'auto';
    updateCharCount();
    
    // Focus sur input
    setTimeout(() => input.focus(), 100);
}

function setQuickMessage(message) {
    const input = document.getElementById('message-input');
    input.value = message;
    autoResizeTextarea(input);
    updateCharCount();
    input.focus();
}

// ==========================================
// INDICATEURS VISUELS
// ==========================================

function showTypingIndicator(agent) {
    hideTypingIndicator(); // Évite duplications
    
    const indicator = document.getElementById('typing-indicator');
    const text = document.getElementById('typing-text');
    
    const agentInfo = AGENTS_INFO[agent?.toLowerCase()] || AGENTS_INFO.anima;
    text.textContent = `${agentInfo.name} réfléchit...`;
    
    indicator.style.display = 'flex';
    AppState.isTyping = true;
    
    // Auto-hide après délai
    setTimeout(() => {
        if (AppState.isTyping) {
            hideTypingIndicator();
        }
    }, CONFIG.typingDelay * 3);
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    indicator.style.display = 'none';
    AppState.isTyping = false;
}

function showTripleStart() {
    showTypingIndicator('triple');
    document.getElementById('typing-text').textContent = '🔥 Débat triangulaire en cours...';
}

function scrollToBottom() {
    const messages = document.getElementById('chat-messages');
    messages.scrollTop = messages.scrollHeight;
}

// ==========================================
// GESTION AGENTS
// ==========================================

function selectAgent(agentName) {
    // Désélection précédent
    document.querySelectorAll('.agent-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Sélection nouveau
    const newBtn = document.querySelector(`[data-agent="${agentName}"]`);
    if (newBtn) {
        newBtn.classList.add('active');
        AppState.selectedAgent = agentName;
        
        // Mise à jour header chat
        updateChatHeader();
        
        console.log('🤖 Agent sélectionné:', agentName);
    }
}

function updateChatHeader() {
    const agentInfo = AGENTS_INFO[AppState.selectedAgent] || AGENTS_INFO.anima;
    
    document.getElementById('selected-agent-icon').textContent = agentInfo.icon;
    document.getElementById('selected-agent-name').textContent = agentInfo.name;
    document.getElementById('selected-agent-desc').textContent = agentInfo.description;
}

// ==========================================
// GESTION DOCUMENTS
// ==========================================

async function uploadDocument(file) {
    // Validation fichier
    if (file.size > CONFIG.maxFileSize) {
        showToast(`Fichier trop volumineux (max ${CONFIG.maxFileSize / 1024 / 1024}MB)`, 'error');
        return;
    }
    
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!CONFIG.supportedFiles.includes(extension)) {
        showToast(`Type de fichier non supporté: ${extension}`, 'error');
        return;
    }
    
    // Progress UI
    const progressEl = document.getElementById('upload-progress');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressEl.style.display = 'block';
    progressText.textContent = 'Upload en cours...';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Animation progress
        progressFill.style.width = '100%';
        progressText.textContent = 'Indexation terminée';
        
        setTimeout(() => {
            progressEl.style.display = 'none';
            progressFill.style.width = '0%';
        }, 1500);
        
        // Mise à jour UI
        AppState.documentsCount++;
        updateDocumentsList(result.filename);
        showToast(`Document "${result.filename}" indexé avec succès`, 'success');
        
        // Refresh stats
        requestSystemStatus();
        
    } catch (error) {
        console.error('❌ Erreur upload:', error);
        showToast(`Erreur upload: ${error.message}`, 'error');
        
        progressEl.style.display = 'none';
        progressFill.style.width = '0%';
    }
}

function updateDocumentsList(filename) {
    const listEl = document.getElementById('documents-list');
    
    const docEl = document.createElement('div');
    docEl.className = 'document-item';
    docEl.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; background: var(--glass-bg); border-radius: 8px; margin: 0.25rem 0;">
            <span>📄</span>
            <span style="flex: 1; font-size: 0.9rem; color: var(--text-secondary);">${escapeHtml(filename)}</span>
            <span style="color: var(--text-muted); font-size: 0.8rem;">✅</span>
        </div>
    `;
    
    listEl.appendChild(docEl);
}

// ==========================================
// GESTION SYSTÈME
// ==========================================

function updateSystemStatus(status) {
    document.getElementById('system-status').textContent = status;
}

function updateSystemStatusFromData(statusData) {
    if (!statusData) return;
    
    AppState.systemStatus = statusData;
    
    // Mise à jour status agents
    if (statusData.agents_status) {
        ['anima', 'neo', 'nexus'].forEach(agent => {
            const statusEl = document.getElementById(`${agent}-status`);
            if (statusEl && statusData.agents_status[agent]) {
                const isAvailable = statusData.agents_status[agent].available;
                statusEl.textContent = isAvailable ? '🟢' : '🔴';
            }
        });
    }
    
    // Mise à jour stats
    if (statusData.database_stats) {
        const dbStats = statusData.database_stats;
        document.getElementById('docs-count').textContent = dbStats.database?.documents_count || 0;
        document.getElementById('interactions-count').textContent = dbStats.database?.interactions_count || 0;
        document.getElementById('vectors-count').textContent = dbStats.vector_store?.total_vectors || 0;
    }
    
    AppState.totalCost = statusData.total_cost || 0;
}

function updateUI() {
    // Mise à jour coûts
    document.getElementById('total-cost').textContent = formatCost(AppState.totalCost);
    document.getElementById('message-count').textContent = AppState.messages.length;
    
    // Mise à jour chunks RAG
    document.getElementById('chunks-value').textContent = AppState.ragChunks;
}

function clearChat() {
    if (confirm('Effacer toute la conversation ?')) {
        AppState.messages = [];
        AppState.totalCost = 0;
        
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-content">
                    <h2>🌟 Conversation effacée</h2>
                    <p>Commence une nouvelle discussion avec tes agents IA !</p>
                </div>
            </div>
        `;
        
        updateUI();
        showToast('Conversation effacée', 'info');
    }
}

function exportChat() {
    if (AppState.messages.length === 0) {
        showToast('Aucun message à exporter', 'info');
        return;
    }
    
    const exportData = {
        session_id: AppState.sessionId,
        timestamp: new Date().toISOString(),
        total_cost: AppState.totalCost,
        selected_agent: AppState.selectedAgent,
        messages: AppState.messages,
        system_status: AppState.systemStatus
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emergence_v4_chat_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
    showToast('Conversation exportée', 'success');
}

// ==========================================
// EVENT LISTENERS
// ==========================================

function setupEventListeners() {
    // Agents selector
    document.querySelectorAll('.agent-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const agent = btn.dataset.agent;
            selectAgent(agent);
        });
    });
    
    // RAG toggle
    const ragToggle = document.getElementById('use-rag');
    ragToggle.addEventListener('change', (e) => {
        AppState.useRag = e.target.checked;
        console.log('🔍 RAG:', AppState.useRag);
    });
    
    // RAG chunks slider
    const ragSlider = document.getElementById('rag-chunks');
    ragSlider.addEventListener('input', (e) => {
        AppState.ragChunks = parseInt(e.target.value);
        document.getElementById('chunks-value').textContent = AppState.ragChunks;
    });
    
    // Upload area
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--anima-color)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--glass-border)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--glass-border)';
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            uploadDocument(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadDocument(e.target.files[0]);
        }
    });
    
    // Message input
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    
    messageInput.addEventListener('input', (e) => {
        updateCharCount();
        autoResizeTextarea(e.target);
    });
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendBtn.addEventListener('click', sendMessage);
    
    // Actions chat
    document.getElementById('clear-chat').addEventListener('click', clearChat);
    document.getElementById('export-chat').addEventListener('click', exportChat);
    
    // Heartbeat WebSocket
    setInterval(() => {
        if (AppState.isConnected && Date.now() - AppState.lastActivity > 30000) {
            requestSystemStatus();
        }
    }, 30000);
}

// ==========================================
// INITIALISATION
// ==========================================

function initializeApp() {
    console.log('🚀 Initialisation ÉMERGENCE V4...');
    
    // État initial
    updateChatHeader();
    updateUI();
    
    // Event listeners
    setupEventListeners();
    
    // Connexion WebSocket
    connectWebSocket();
    
    // Focus input
    setTimeout(() => {
        document.getElementById('message-input').focus();
    }, 500);
    
    console.log('✅ ÉMERGENCE V4 initialisée !');
    showToast('🧠 ÉMERGENCE V4 prêt !', 'success');
}

// Démarrage dès que le DOM est prêt
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Export global pour debug
window.EmergenceV4 = {
    AppState,
    selectAgent,
    sendMessage,
    setQuickMessage,
    requestSystemStatus,
    uploadDocument,
    clearChat,
    exportChat
};